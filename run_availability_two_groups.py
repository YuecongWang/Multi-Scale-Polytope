"""Reconstructed CIFAR-10 availability controls, not the lost original pipeline."""
import argparse, os, json, hashlib, time, random, csv, sys
from pathlib import Path
from collections import Counter
p=argparse.ArgumentParser()
p.add_argument('--stage',choices=['generate','train','all'],default='all')
p.add_argument('--data',default='datasets/CIFAR10_TRAIN_Split.pth')
p.add_argument('--out',default='availability_controls_v1')
p.add_argument('--gpu',default='0')
p.add_argument('--target-label',type=int,default=6)
p.add_argument('--base-label',type=int,default=8)
p.add_argument('--proxy',default='ResNet50')
p.add_argument('--proxy-checkpoint',default='ckpt-%s-4800-dp0.200-droplayer0.000-seed1226.t7')
p.add_argument('--proxy-dropout',type=float,default=0.2)
p.add_argument('--generation-seed',type=int,default=2026)
p.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
p.add_argument('--victims',nargs='+',default=['ResNet50'])
p.add_argument('--epochs',type=int,default=100)
p.add_argument('--batch-size',type=int,default=64)
p.add_argument('--max-groups',type=int,default=48)
a=p.parse_args()
if not 1<=a.max_groups<=48: p.error('max-groups must be within 1..48')
if not 0<=a.base_label<10 or not 0<=a.target_label<10 or a.base_label==a.target_label:
    p.error('Choose different CIFAR-10 target/base labels')
os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import models
from utils import load_pretrained_net
if not torch.cuda.is_available(): raise RuntimeError('CUDA required')

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def save_torch(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,temp);temp.replace(path)

out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,s):
        for stream in self.streams:stream.write(s)
    def flush(self):
        for stream in self.streams:stream.flush()
log=(out/'run.log').open('a',encoding='utf-8',buffering=1)
sys.stdout=Tee(sys.stdout,log)
sys.stderr=Tee(sys.stderr,log)
mean=torch.tensor([.4914,.4822,.4465]).view(1,3,1,1)
std=torch.tensor([.2023,.1994,.2010]).view(1,3,1,1)
totensor=transforms.ToTensor()
data=torch.load(a.data,map_location='cpu',weights_only=False)
assert len(data['clean_train'])==48000 and len(data['others'])==2000
for subset,count in [('clean_train',4800),('others',200)]:
    assert Counter(int(y) for _,y in data[subset])==Counter({i:count for i in range(10)})
clean=[(totensor(x),int(y)) for x,y in data['others']]
base_candidates=[i for i,(_,y) in enumerate(data['clean_train']) if int(y)==a.base_label]
target_candidates=[i for i,(_,y) in enumerate(data['clean_train']) if int(y)==a.target_label]
rng=np.random.RandomState(a.generation_seed)
base_indices=rng.permutation(base_candidates)[:480].tolist()
target_indices=rng.permutation(target_candidates).tolist()
proxy_path=Path('model-chks')/(a.proxy_checkpoint % a.proxy)
protocol=dict(protocol='reconstructed_availability_squared_two_groups_v2',data_sha256=sha(a.data),
    data_path=str(Path(a.data).resolve()),target_label=a.target_label,base_label=a.base_label,
    generation_seed=a.generation_seed,base_indices=base_indices,target_groups=[target_indices[i*100:(i+1)*100] for i in range(48)],
    poison_selection=list(range(400)),clean_subset='others',clean_count=2000,
    groups=48,targets_per_group=100,poisons_per_group=10,poison_count_used=400,
    poison_iterations=800,poison_lr=.04,epsilon=.1,feature='three-stage Flat concatenation',
    objective='half normalized squared centroid residual; single proxy',proxy=a.proxy,proxy_dropout=a.proxy_dropout,
    proxy_checkpoint=str(proxy_path.resolve()),proxy_sha256=sha(proxy_path) if proxy_path.exists() else None,
    randomness='fixed poison generation; paired random victim initializations and training seeds',
    victim_initialization='from scratch, no pretrained victim',training_epochs=a.epochs,batch_size=a.batch_size,
    optimizer='SGD lr0.1 momentum0.9 weight_decay5e-4; cosine decay',
    augmentation='RandomCrop32 padding4; RandomHorizontalFlip',
    poison_ratio_over_clean=.2,poison_fraction_of_augmented_training=400/2400,
    gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,torchvision_version=torchvision.__version__)
manifest=out/'protocol.json'
if manifest.exists():
    previous=json.loads(manifest.read_text())
    if previous!=protocol:raise RuntimeError('Protocol changed: use a different --out directory')
else:save(manifest,protocol)
print(json.dumps(protocol,ensure_ascii=False,indent=2));sys.stdout.flush()

if a.stage in ['generate','all']:
    if not proxy_path.is_file():raise FileNotFoundError(proxy_path)
    seed(a.generation_seed)
    net=load_pretrained_net(a.proxy,a.proxy_checkpoint,'model-chks',test_dp=a.proxy_dropout)
    net.eval()
    for param in net.parameters():param.requires_grad_(False)
    model=net.module
    for group in range(a.max_groups):
        checkpoint=out/f'group_{group:02d}.pth'
        if checkpoint.exists():
            print('[SKIP GENERATION]',group);continue
        seed(a.generation_seed+group)
        base=torch.stack([totensor(data['clean_train'][i][0]) for i in base_indices[group*10:(group+1)*10]]).cuda()
        device_mean=mean.cuda();device_std=std.cuda()
        with torch.no_grad():
            feature_sum=None
            ids=protocol['target_groups'][group]
            for start in range(0,100,10):
                batch=torch.stack([totensor(data['clean_train'][i][0]) for i in ids[start:start+10]]).cuda()
                features=model.penultimate2((batch-device_mean)/device_std)
                part=features.sum(0,keepdim=True)
                feature_sum=part if feature_sum is None else feature_sum+part
            target_feature=feature_sum/100
            norm=target_feature.norm()
            if norm.item()==0:raise RuntimeError('Zero target feature norm')
        # Original attack lr is applied in normalized pixel coordinates.
        normalized=torch.nn.Parameter(((base-device_mean)/device_std).clone())
        optimizer=torch.optim.Adam([normalized],lr=.04)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        diagnostics=[]
        for iteration in range(800):
            optimizer.zero_grad()
            features=model.penultimate2(normalized)
            loss=0.5*(features.mean(0,keepdim=True)-target_feature).square().sum()/norm.square()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();optimizer.step()
            with torch.no_grad():
                pixels_now=normalized*device_std+device_mean
                bounded=base+(pixels_now-base).clamp(-.1,.1)
                normalized.copy_((bounded.clamp(0,1)-device_mean)/device_std)
            if iteration%50==0 or iteration==799:
                record=dict(iteration=iteration,loss=float(loss.detach()))
                diagnostics.append(record);print('[GEN]',group,record);sys.stdout.flush()
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        final=(normalized.detach()*device_std+device_mean).clamp(0,1)
        delta=final-base
        assert float(delta.abs().max())<=.100001
        metadata=dict(group=group,seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),actual_linf=float(delta.abs().max()),
            per_image_l2=delta.flatten(1).norm(dim=1).cpu().tolist(),diagnostics=diagnostics,
            timing_scope='optimization only; excludes proxy loading and target centroid extraction')
        save_torch(checkpoint,dict(images=final.cpu(),base_images=base.cpu(),label=a.base_label,
            base_indices=base_indices[group*10:(group+1)*10],metadata=metadata))
        save(out/f'group_{group:02d}.json',metadata)
        del normalized,optimizer,features,base,target_feature,feature_sum,final,delta,bounded,pixels_now
        torch.cuda.empty_cache()
    del net,model;torch.cuda.empty_cache()

if a.stage in ['train','all']:
    paths=[out/f'group_{i:02d}.pth' for i in range(48)]
    if not all(path.exists() for path in paths):
        raise RuntimeError('Generate all 48 groups before training (omit --max-groups 1)')
    batches=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    poisoned=torch.cat([batch['images'] for batch in batches])[:400]
    originals=torch.cat([batch['base_images'] for batch in batches])[:400]
    seed(a.generation_seed+100000)
    random_images=(originals+(torch.rand_like(originals)*2-1)*.1).clamp(0,1)
    modes=['clean','msp']
    pool={ 'unperturbed':originals,'random':random_images,'msp':poisoned }
    aug=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip()])
    normalize=transforms.Normalize(mean.flatten().tolist(),std.flatten().tolist())
    class TrainingSet(Dataset):
        def __init__(self,mode):self.mode=mode;self.extra=pool.get(mode)
        def __len__(self):return 2000+(0 if self.extra is None else 400)
        def __getitem__(self,index):
            if index<2000:x,y=clean[index]
            else:x,y=self.extra[index-2000],a.base_label
            return normalize(aug(x)),y
    testset=torchvision.datasets.CIFAR10('datasets',train=False,download=True,
        transform=transforms.Compose([transforms.ToTensor(),normalize]))
    testloader=DataLoader(testset,batch_size=128,shuffle=False,num_workers=0)
    @torch.no_grad()
    def evaluate(net):
        net.eval();correct=torch.zeros(10,dtype=torch.long);total=correct.clone()
        for x,y in testloader:
            pred=net(x.cuda()).argmax(1).cpu()
            for label in range(10):
                mask=y==label;total[label]+=mask.sum();correct[label]+=((pred==y)&mask).sum()
        return dict(accuracy=100*int(correct.sum())/int(total.sum()),
            per_class_accuracy=[100*int(c)/int(n) for c,n in zip(correct,total)])
    for victim in a.victims:
        if not hasattr(models,victim):raise ValueError('Unsupported project model '+victim)
        for training_seed in a.seeds:
            seed(training_seed)
            initial=models.__dict__[victim]()
            initial_state={k:v.detach().cpu().clone() for k,v in initial.state_dict().items()};del initial
            for mode in modes:
                directory=out/victim/f'seed{training_seed}'/mode;directory.mkdir(parents=True,exist_ok=True)
                result_path=directory/'result.json'
                if result_path.exists():print('[SKIP TRAIN]',victim,training_seed,mode);continue
                seed(training_seed);net=models.__dict__[victim]();net.load_state_dict(initial_state);net.cuda()
                # Reset RNG after construction for paired sampler/augmentation streams.
                seed(training_seed)
                dset=TrainingSet(mode)
                loader=DataLoader(dset,batch_size=a.batch_size,shuffle=True,num_workers=0)
                optimizer=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9,weight_decay=5e-4)
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs)
                metadata=dict(victim=victim,seed=training_seed,mode=mode,training_count=len(dset),
                    class_counts={str(i):200+(400 if mode!='clean' and i==a.base_label else 0) for i in range(10)},
                    protocol_path=str(manifest.resolve()),initialization='paired random initialization',
                    all_parameters_updated=True,epochs=a.epochs)
                print(json.dumps(metadata,indent=2));torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter();history=[]
                for epoch in range(a.epochs):
                    net.train();loss_sum=0;count=0;train_correct=0
                    for x,y in loader:
                        x,y=x.cuda(),y.cuda();optimizer.zero_grad();output=net(x)
                        loss=torch.nn.functional.cross_entropy(output,y);loss.backward();optimizer.step()
                        loss_sum+=float(loss.detach())*len(y);count+=len(y);train_correct+=int((output.argmax(1)==y).sum())
                    scheduler.step()
                    record=dict(epoch=epoch,loss=loss_sum/count,train_accuracy=100*train_correct/count)
                    history.append(record)
                    if epoch%10==0 or epoch==a.epochs-1:print('[TRAIN]',victim,training_seed,mode,record);sys.stdout.flush()
                torch.cuda.synchronize();train_seconds=time.perf_counter()-started
                peak_allocated=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
                metrics=evaluate(net)
                save(result_path,dict(metadata,**metrics,train_seconds=train_seconds,
                    peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,
                    cost_scope='victim training only, excludes test evaluation',history=history,status='completed'))
                save_torch(directory/'final_model.pth',dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},**metadata))
                print('[DONE]',victim,training_seed,mode,metrics);del net,optimizer,scheduler;torch.cuda.empty_cache()
    rows=[]
    for path in out.glob('*/seed*/*/result.json'):
        r=json.loads(path.read_text());row={k:r[k] for k in ['victim','seed','mode','training_count','accuracy','train_seconds','peak_allocated_bytes','peak_reserved_bytes']}
        row.update({f'class_{i}_accuracy':value for i,value in enumerate(r['per_class_accuracy'])});rows.append(row)
    if rows:
        with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Saved controls:',out.resolve())
