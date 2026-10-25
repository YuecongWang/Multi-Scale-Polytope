"""Run from the existing Multi-scalePolytope folder. Never crafts poisons."""
import argparse, csv, json, os, random, sys, time
from experiment_recorder import digest, save_json
from pathlib import Path
from collections import Counter
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument('--start', type=int, default=0)
p.add_argument('--end', type=int, default=49)
p.add_argument('--seed', type=int, default=2026)
p.add_argument('--gpu', default='0')
p.add_argument('--config', default='experiments.json')
p.add_argument('--victim-nets', nargs='+', default=['ResNet50'])
p.add_argument('--methods', nargs='+', default=None)
a = p.parse_args()
if not 0 <= a.start <= a.end <= 49: p.error('Target range must be within 0..49')
os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
import numpy as np
import torch
import torchvision
from torchvision import transforms
from utils import load_pretrained_net, fetch_target
from dataloader import PoisonedDataset
from trainer import train_network_with_poison

project = Path.cwd()
root = project.parent
config = json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
checkpoint_template = config.get('victim_checkpoint_template', 'ckpt-%s-4800.t7')
model_dir = config.get('model_directory', 'model-chks')
checkpoint = project / model_dir / (checkpoint_template % a.victim_nets[0])
datafile = project / config.get('train_data_path', 'datasets/CIFAR10_TRAIN_Split.pth')
for f in (checkpoint, datafile):
    if not f.is_file(): raise FileNotFoundError(f)
if not torch.cuda.is_available(): raise RuntimeError('CUDA required')


def seed():
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Tee:
    def __init__(self, terminal, file): self.terminal, self.file = terminal, file
    def write(self, s): self.terminal.write(s); self.file.write(s)
    def flush(self): self.terminal.flush(); self.file.flush()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
    (0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
testset = torchvision.datasets.CIFAR10(root='datasets', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False)

@torch.no_grad()
def measure(net, target):
    logits = net(target.cuda()).flatten()
    prediction = int(logits.argmax())
    probabilities = logits.softmax(0).cpu().tolist()
    correct = torch.zeros(10, dtype=torch.long)
    totals = torch.zeros(10, dtype=torch.long)
    for images, labels in loader:
        preds = net(images.cuda()).argmax(1).cpu()
        for cls in range(10):
            mask = labels == cls
            totals[cls] += mask.sum()
            correct[cls] += ((preds == labels) & mask).sum()
    return dict(prediction=prediction, logits=logits.cpu().tolist(), probabilities=probabilities,
        clean_accuracy_percent=100*int(correct.sum())/int(totals.sum()),
        class_accuracy_percent=[100*int(c)/int(n) if n else None for c,n in zip(correct,totals)])

methods = {}
for name, relative in config['methods'].items():
    if a.methods is not None and name not in a.methods: continue
    folder = Path(relative)
    methods[name] = folder if folder.is_absolute() else project / folder

if not methods: raise ValueError('No methods selected')
if a.methods and set(a.methods) - set(methods): raise ValueError('Unknown method selected')
for victim in a.victim_nets:
    if not (project/model_dir/(checkpoint_template % victim)).is_file():
        raise FileNotFoundError(project/model_dir/(checkpoint_template % victim))
for method, folder in methods.items():
    for target in range(a.start, a.end+1):
        path=folder/str(target)/'poison_00799.pth'
        if not path.is_file(): raise FileNotFoundError(path)

out = project/config.get('output_directory', 'evaluation_recorded')/f'seed{a.seed}'
out.mkdir(parents=True, exist_ok=True)
identity = dict(victim_checkpoint=str(checkpoint), victim_checkpoint_sha256=digest(checkpoint),
    data_path=str(datafile), data_sha256=digest(datafile), overlap=config.get('overlap', 'unknown_not_verified'),
    torch_version=torch.__version__, torchvision_version=torchvision.__version__,
    gpu_name=torch.cuda.get_device_name(0), evaluation_seed=a.seed,
    randomness='Reseed before each independent victim evaluation; no poison generation',
    training='linear transfer: classifier head only; net.eval()',
    proxy_ensemble='Historical generation: 18 instances; no proxy loaded in this evaluation')
args = SimpleNamespace(end2end=False, retrain_opt='adam', retrain_lr=0.1,
    retrain_wd=0, retrain_momentum=0.9, retrain_bsize=64, retrain_epochs=60,
    lr_decay_epoch=[30,45], train_data_path=str(datafile), num_per_class=50,
    subset_group=0, dset_path='datasets', device='cuda', target_label=6, poison_label=8)

for victim in a.victim_nets:
    checkpoint = project / model_dir / (checkpoint_template % victim)
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    identity.update(victim_checkpoint=str(checkpoint), victim_checkpoint_sha256=digest(checkpoint))
    for target_index in range(a.start,a.end+1):
        for method, folder in methods.items():
            poison_path=folder/str(target_index)/'poison_00799.pth'
            directory=out/method/victim/str(target_index); directory.mkdir(parents=True,exist_ok=True)
            result_path=directory/'result.json'
            if result_path.exists():
                old=json.loads(result_path.read_text())
                if (old.get('status')=='completed' and old.get('victim_checkpoint_sha256')==identity['victim_checkpoint_sha256']
                    and old.get('data_sha256')==identity['data_sha256'] and old.get('evaluation_seed')==a.seed
                    and old.get('poison_sha256')==digest(poison_path)):
                    print('[SKIP]',method,target_index); continue
                raise RuntimeError(f'Existing result provenance differs: {result_path}')
            terminal=sys.stdout
            with (directory/'log.txt').open('w',encoding='utf-8') as log:
                sys.stdout=Tee(terminal,log)
                try:
                    state=torch.load(str(poison_path),map_location='cpu',weights_only=False)
                    poisons,indices=state['poison'],state['idx']
                    target=fetch_target(6,target_index,50,subset='others',path=str(datafile),transforms=transform)
                    dset=PoisonedDataset(str(datafile),subset='others',transform=transform,
                        num_per_label=50,poison_tuple_list=poisons,poison_indices=indices,subset_group=0)
                    labels=[int(label) for _,label in poisons]
                    labels += [int(dset.img_label_list[i][1]) for i in dset.valid_indices]
                    meta=dict(identity,method=method,victim=victim,target_index=target_index,target_label=6,poison_label=8,
                        poison_path=str(poison_path),poison_sha256=digest(poison_path),
                        generation_seed='2026_as_reported_in_source_logs_not_recovered_from_checkpoint',
                        base_indices=[int(i) for i in indices],poison_count=len(poisons),
                        actual_training_count=len(dset),class_counts=dict(Counter(labels)),
                        poison_fraction=len(poisons)/len(dset),training_parameters=vars(args))
                    print(json.dumps(meta,ensure_ascii=False,indent=2))
                    seed(); net=load_pretrained_net(victim, checkpoint_template, model_dir)
                    before=measure(net,target)
                    # Reset after baseline measurements so they do not consume training RNG.
                    seed(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    started=time.perf_counter()
                    prediction=train_network_with_poison(net,target,poisons,indices,str(directory),args,save_state=False)
                    torch.cuda.synchronize()
                    seconds=time.perf_counter()-started
                    allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
                    after=measure(net,target)
                    if prediction != after['prediction']: raise RuntimeError('Post-training prediction mismatch')
                    result=dict(meta,status='completed',before=before,after=after,
                        attack_success=int(prediction==8),baseline_target_correct=int(before['prediction']==6),
                        victim_training_and_builtin_evaluation_seconds=seconds,
                        peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,
                        cost_scope='After victim load/baseline; includes training and trainer built-in clean evaluation; excludes extra post metrics and generation')
                    print(json.dumps(result,ensure_ascii=False,indent=2))
                    save_json(result_path, result)
                    del net,poisons
                finally: sys.stdout=terminal
            torch.cuda.empty_cache()

rows=[]
for method in methods:
    for file in sorted((out/method).glob('*/*/result.json')):
        r=json.loads(file.read_text())
        if r.get('status')!='completed': continue
        row=dict(method=method,victim=r['victim'],target_index=r['target_index'],success=r['attack_success'],
            baseline_target_correct=r['baseline_target_correct'],before_prediction=r['before']['prediction'],
            after_prediction=r['after']['prediction'],before_clean_accuracy=r['before']['clean_accuracy_percent'],
            after_clean_accuracy=r['after']['clean_accuracy_percent'],training_count=r['actual_training_count'],
            poison_fraction=r['poison_fraction'],seconds=r['victim_training_and_builtin_evaluation_seconds'],
            peak_allocated_MiB=r['peak_allocated_bytes']/2**20,peak_reserved_MiB=r['peak_reserved_bytes']/2**20)
        for cls in range(10): row[f'class_{cls}_accuracy']=r['after']['class_accuracy_percent'][cls]
        rows.append(row)
if rows:
    with (out/'per_target.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Saved:',out)
for method in methods:
    for victim in a.victim_nets:
        selected=[r for r in rows if r['method']==method and r['victim']==victim]
        if selected: print(method,victim,len(selected),'targets; ASR',100*sum(r['success'] for r in selected)/len(selected))
