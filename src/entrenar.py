import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1") # comparación reproducible en CPU
import json, time, platform, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from IPython.display import display, Markdown
SEED, EPOCHS, BATCH_SIZE = 42, 20, 32
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
candidates = [Path.cwd(), Path.cwd().parent, Path('/content/Evaluacion2_001D')]
ROOT = next((p for p in candidates if (p/'data/sign_mnist_train.csv').exists()), None)
if ROOT is None:
    raise FileNotFoundError("Descomprima el proyecto completo y abra el notebook desde notebooks/ o la raíz.")
for d in ['models','images','results']: (ROOT/d).mkdir(exist_ok=True)
LETTERS = ['A','C','F','K','O','Y']
ORIGINAL = [0,2,5,10,14,24]
MAPPING = dict(zip(ORIGINAL, range(6)))
plt.rcParams.update({'figure.dpi':120, 'font.size':11})
def savefig(name):
    plt.tight_layout(); plt.savefig(ROOT/'images'/name, dpi=160, bbox_inches='tight'); plt.show(); plt.close()
environment = {'python':sys.version, 'tensorflow':tf.__version__, 'numpy':np.__version__, 'platform':platform.platform(), 'processor':platform.processor(), 'devices':[str(d) for d in tf.config.list_physical_devices()], 'seed':SEED, 'epochs':EPOCHS, 'batch_size':BATCH_SIZE, 'threads':2, 'oneDNN':os.environ['TF_ENABLE_ONEDNN_OPTS']}
(ROOT/'results/environment.json').write_text(json.dumps(environment,indent=2))
print(json.dumps(environment,indent=2))

train_raw = pd.read_csv(ROOT/'data/sign_mnist_train.csv')
test_raw = pd.read_csv(ROOT/'data/sign_mnist_test.csv')
quality = {}
for name, frame in [('train_original',train_raw),('test_original',test_raw)]:
    assert frame.shape[1] == 785 and frame.columns[0]=='label'
    assert frame.isna().sum().sum()==0
    assert frame.iloc[:,1:].to_numpy().min()>=0 and frame.iloc[:,1:].to_numpy().max()<=255
    quality[name]={'rows':len(frame),'columns':frame.shape[1],'nulls':int(frame.isna().sum().sum()), 'min_pixel':int(frame.iloc[:,1:].to_numpy().min()),'max_pixel':int(frame.iloc[:,1:].to_numpy().max())}
train = train_raw[train_raw.label.isin(ORIGINAL)].copy()
test = test_raw[test_raw.label.isin(ORIGINAL)].copy()
y = train.label.map(MAPPING).to_numpy(dtype='int64')
yt = test.label.map(MAPPING).to_numpy(dtype='int64')
X = train.iloc[:,1:].to_numpy(dtype='float32')/255.0
Xt = test.iloc[:,1:].to_numpy(dtype='float32')/255.0
itr, iva = train_test_split(np.arange(len(y)),test_size=0.20,random_state=SEED,stratify=y)
Xtr,Xva,ytr,yva=X[itr],X[iva],y[itr],y[iva]
CTr,CVa,CTe=[a.reshape(-1,28,28,1) for a in [Xtr,Xva,Xt]]
assert set(itr).isdisjoint(set(iva)) and len(itr)+len(iva)==len(y)
assert (len(ytr),len(yva),len(yt))==(5521,1381,1797)
assert np.array_equal(CTr.reshape(-1,784),Xtr)
np.savez(ROOT/'results/split_indices.npz',train_filtered=itr,validation_filtered=iva,train_csv_rows=train.index.to_numpy()[itr],validation_csv_rows=train.index.to_numpy()[iva],test_csv_rows=test.index.to_numpy())
def hashes(frame):
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in frame.iloc[:,1:].to_numpy(dtype='uint8')]
ht,hs=hashes(train),hashes(test)
quality['duplicates_filtered_train']=len(ht)-len(set(ht))
quality['duplicates_filtered_test']=len(hs)-len(set(hs))
quality['train_test_exact_overlap']=len(set(ht)&set(hs))
quality['train_validation_exact_overlap']=len(set(np.array(ht)[itr])&set(np.array(ht)[iva]))
quality['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.csv')}
(ROOT/'results/data_quality.json').write_text(json.dumps(quality,indent=2))
distribution=pd.DataFrame({'Letra':LETTERS,'Original':ORIGINAL,'Codificada':range(6),'Train':np.bincount(ytr,minlength=6),'Validacion':np.bincount(yva,minlength=6),'Test':np.bincount(yt,minlength=6)})
distribution.to_csv(ROOT/'results/class_distribution.csv',index=False)
display(distribution); print(json.dumps(quality,indent=2))
fig,axs=plt.subplots(1,6,figsize=(12,2.8))
for k,ax in enumerate(axs):
    ax.imshow(CTr[np.flatnonzero(ytr==k)[0],:,:,0],cmap='gray',vmin=0,vmax=1);ax.set_title(f'{LETTERS[k]} · clase {k}');ax.axis('off')
savefig('clases.png')
distribution.set_index('Letra')[['Train','Validacion','Test']].plot.bar(figsize=(9,4));plt.ylabel('Imágenes');plt.xticks(rotation=0);savefig('distribucion.png')

L=tf.keras.layers
models={}
histories={}
times={}
def dataset(x,y,training=False):
    ds=tf.data.Dataset.from_tensor_slices((x,y))
    if training: ds=ds.shuffle(len(y),seed=SEED,reshuffle_each_iteration=True)
    ds=ds.batch(BATCH_SIZE)
    opt=tf.data.Options();opt.threading.private_threadpool_size=2;opt.threading.max_intra_op_parallelism=2
    return ds.with_options(opt)
for name,a,b in [('MLP',Xtr,Xva),('CNN',CTr,CVa)]:
    tf.keras.utils.set_random_seed(SEED)
    if name=='MLP':
        model=tf.keras.Sequential([L.Input((784,)),L.Dense(128,activation='relu'),L.Dense(64,activation='relu'),L.Dense(6,activation='softmax')],name='MLP')
    else:
        model=tf.keras.Sequential([L.Input((28,28,1)),L.Conv2D(16,3,activation='relu',name='conv1'),L.MaxPooling2D(2,name='pool1'),L.Conv2D(32,3,activation='relu',name='conv2'),L.MaxPooling2D(2,name='pool2'),L.Flatten(),L.Dense(64,activation='relu',name='dense64'),L.Dense(6,activation='softmax',name='salida')],name='CNN')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    start=time.perf_counter()
    hist=model.fit(dataset(a,ytr,True),validation_data=dataset(b,yva),epochs=EPOCHS,verbose=2)
    times[name]=time.perf_counter()-start
    models[name]=model;histories[name]=hist.history
    model.save(ROOT/'models'/f'{name.lower()}.keras')
    pd.DataFrame(hist.history,index=np.arange(1,EPOCHS+1)).rename_axis('epoch').to_csv(ROOT/'results'/f'history_{name.lower()}.csv')
assert models['MLP'].count_params()==109126
assert models['CNN'].count_params()==56454

fig,axs=plt.subplots(2,2,figsize=(12,7))
for row,name in enumerate(['MLP','CNN']):
    h=histories[name]
    for col,key in enumerate(['loss','accuracy']):
        axs[row,col].plot(range(1,EPOCHS+1),h[key],label='Train')
        axs[row,col].plot(range(1,EPOCHS+1),h['val_'+key],label='Validación')
        axs[row,col].set(title=f'{name} · {key}',xlabel='Época',ylabel=key);axs[row,col].legend();axs[row,col].grid(alpha=.2)
savefig('curvas.png')
for name,h in histories.items():
    print(name, 'Tiempo (s):',round(times[name],3),'época de menor val_loss:',1+int(np.argmin(h['val_loss'])), 'última val_loss:',h['val_loss'][-1])

rows=[];probs={};preds={};reports={};cms={}
for name,tr,va,te in [('MLP',Xtr,Xva,Xt),('CNN',CTr,CVa,CTe)]:
    model=models[name]
    probs[name]=model.predict(dataset(te,yt),verbose=0)
    preds[name]=probs[name].argmax(1)
    reports[name]=classification_report(yt,preds[name],target_names=LETTERS,output_dict=True,zero_division=0)
    pd.DataFrame(reports[name]).T.to_csv(ROOT/'results'/f'report_{name.lower()}.csv')
    pd.DataFrame({'csv_row':test.index,'true':yt,'predicted':preds[name],'true_letter':np.array(LETTERS)[yt],'predicted_letter':np.array(LETTERS)[preds[name]],'probability':probs[name].max(1)}).to_csv(ROOT/'results'/f'predictions_{name.lower()}.csv',index=False)
    for _ in range(10):model(te[:1],training=False).numpy()
    latency=[]
    for _ in range(100):
        t=time.perf_counter();model(te[:1],training=False).numpy();latency.append((time.perf_counter()-t)*1000)
    evals={split:model.evaluate(dataset(x,z),verbose=0) for split,x,z in [('train',tr,ytr),('val',va,yva),('test',te,yt)]}
    row={'model':name,'parameters':model.count_params(),'fit_seconds':times[name],'errors':int((preds[name]!=yt).sum()),'latency_median_ms':float(np.median(latency)),'latency_p95_ms':float(np.percentile(latency,95))}
    for split,values in evals.items():row[split+'_loss'],row[split+'_accuracy']=values
    for avg in ['macro','weighted']:
        for label,fun in [('precision',precision_score),('recall',recall_score),('f1',f1_score)]: row[label+'_'+avg]=fun(yt,preds[name],average=avg,zero_division=0)
    row['train_test_gap_pp']=100*(row['train_accuracy']-row['test_accuracy'])
    row['train_val_gap_pp']=100*(row['train_accuracy']-row['val_accuracy'])
    rows.append(row)
    cms[name]=confusion_matrix(yt,preds[name],labels=range(6))
    pd.DataFrame(cms[name],index=LETTERS,columns=LETTERS).to_csv(ROOT/'results'/f'confusion_{name.lower()}.csv')
    print(name);display(pd.DataFrame(reports[name]).T.round(4))
metrics=pd.DataFrame(rows).set_index('model');metrics.to_csv(ROOT/'results/metrics.csv')
(ROOT/'results/metrics.json').write_text(metrics.to_json(orient='index',indent=2))
display(metrics.round(4))
fig,axs=plt.subplots(2,2,figsize=(12,10))
for i,name in enumerate(['MLP','CNN']):
    cm=cms[name]
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=LETTERS,yticklabels=LETTERS,ax=axs[i,0],cbar=False)
    sns.heatmap(cm/cm.sum(1,keepdims=True)*100,annot=True,fmt='.1f',cmap='Blues',vmin=0,vmax=100,xticklabels=LETTERS,yticklabels=LETTERS,ax=axs[i,1],cbar=False)
    for j in range(2):axs[i,j].set(xlabel='Predicha',ylabel='Real',title=f'{name} · '+('cantidades' if j==0 else '% por clase real'))
savefig('matrices.png')

historical=pd.DataFrame([{'modelo':'MLP histórico E1','accuracy':.8703,'precision_macro':.8778,'recall_macro':.8698,'f1_macro':.8694,'parameters':109126,'errors':233}])
historical.to_csv(ROOT/'results/historical_e1.csv',index=False)
display(historical)
fig,axs=plt.subplots(1,3,figsize=(13,4))
metrics[['test_accuracy','precision_macro','recall_macro','f1_macro']].T.plot.bar(ax=axs[0],rot=20,ylim=(0,1.08));axs[0].set_title('Métricas de test')
metrics.parameters.plot.bar(ax=axs[1],rot=0,color=['#64748b','#0f766e']);axs[1].set_title('Parámetros')
metrics.fit_seconds.plot.bar(ax=axs[2],rot=0,color=['#64748b','#0f766e']);axs[2].set_title('Entrenamiento (s)')
savefig('comparacion.png')
delta=100*(metrics.loc['CNN','test_accuracy']-metrics.loc['MLP','test_accuracy'])
print(f'Diferencia CNN − MLP actual: {delta:.2f} puntos porcentuales.')
print(f'CNN usa {100*(1-56454/109126):.2f}% menos parámetros.')
for name in metrics.index:
    m=metrics.loc[name]
    print(f"{name}: brecha train-test {m.train_test_gap_pp:.2f} pp; accuracy≥90%: {m.test_accuracy>=.9}; F1 macro≥0,88: {m.f1_macro>=.88}; error<5%: {1-m.test_accuracy<.05}; latencia p95<10 ms: {m.latency_p95_ms<10}")

confusions=[]
for name in ['MLP','CNN']:
    cm=cms[name]
    for i in range(6):
        for j in range(6):
            if i!=j and cm[i,j]>0:confusions.append({'model':name,'real':LETTERS[i],'predicted':LETTERS[j],'count':int(cm[i,j]),'percent_of_class':100*cm[i,j]/cm[i].sum()})
    fig,axs=plt.subplots(2,6,figsize=(13,5))
    for row,correct in enumerate([True,False]):
        ids=np.flatnonzero((preds[name]==yt)==correct)[:6]
        for j,ax in enumerate(axs[row]):
            ax.axis('off')
            if j<len(ids):
                ix=ids[j];ax.imshow(CTe[ix,:,:,0],cmap='gray');ax.set_title(f'fila {test.index[ix]}\n{LETTERS[yt[ix]]} → {LETTERS[preds[name][ix]]}\np={probs[name][ix].max():.3f}',fontsize=10,color='#0f766e' if correct else '#b91c1c')
            elif j==0:ax.text(.1,.5,'Sin errores' if not correct else 'Sin aciertos')
    plt.suptitle(f'{name}: aciertos arriba, errores abajo');savefig(f'ejemplos_{name.lower()}.png')
confusion_table=pd.DataFrame(confusions,columns=['model','real','predicted','count','percent_of_class']).sort_values('count',ascending=False)
confusion_table.to_csv(ROOT/'results/error_pairs.csv',index=False);display(confusion_table)
for name in ['MLP','CNN']:
    ok=preds[name]==yt
    print(name,'probabilidad media en aciertos:',probs[name][ok].max(1).mean(), 'en errores:',probs[name][~ok].max(1).mean() if (~ok).any() else 'no aplica')

cnn=models['CNN']
feature_model=tf.keras.Model(cnn.inputs,[cnn.get_layer('conv1').output,cnn.get_layer('conv2').output])
representatives=[]
for k in range(6):
    ids=np.flatnonzero((yt==k)&(preds['CNN']==k))
    representatives.append(int(ids[0] if len(ids) else np.flatnonzero(yt==k)[0]))
f1,f2=feature_model(CTe[representatives],training=False)
fig,axs=plt.subplots(6,9,figsize=(15,10))
for r,ix in enumerate(representatives):
    axs[r,0].imshow(CTe[ix,:,:,0],cmap='gray');axs[r,0].set_title(f'{LETTERS[r]} · fila {test.index[ix]}',fontsize=9)
    for k in range(4):
        axs[r,k+1].imshow(f1[r,:,:,k],cmap='viridis');axs[r,k+1].set_title(f'Conv1 / {k}',fontsize=9)
        axs[r,k+5].imshow(f2[r,:,:,k],cmap='viridis');axs[r,k+5].set_title(f'Conv2 / {k}',fontsize=9)
for ax in axs.flat:ax.axis('off')
savefig('feature_maps.png')
np.savez_compressed(ROOT/'results/feature_maps.npz',conv1=f1.numpy(),conv2=f2.numpy(),test_filtered_indices=representatives)
print('Fracción de ceros por capa:',float(np.mean(f1.numpy()==0)),float(np.mean(f2.numpy()==0)))

grad_model=tf.keras.Model(cnn.inputs,[cnn.get_layer('conv2').output,cnn.get_layer('salida').input])
last=cnn.get_layer('salida')
heats=[]
fig,axs=plt.subplots(2,6,figsize=(13,5))
for k,ix in enumerate(representatives):
    image=tf.convert_to_tensor(CTe[ix:ix+1])
    with tf.GradientTape() as tape:
        conv,features=grad_model(image,training=False)
        logits=tf.matmul(features,last.kernel)+last.bias
        score=logits[:,k]
    gradients=tape.gradient(score,conv)
    assert gradients is not None
    weights=tf.reduce_mean(gradients,axis=(0,1,2))
    heat=tf.nn.relu(tf.reduce_sum(conv[0]*weights,axis=-1))
    maximum=float(tf.reduce_max(heat))
    heat=heat/(maximum+1e-10)
    heat=tf.image.resize(heat[...,None],(28,28)).numpy()[:,:,0]
    heats.append(heat)
    axs[0,k].imshow(CTe[ix,:,:,0],cmap='gray');axs[0,k].set_title(f'Real {LETTERS[k]}\nPred. {LETTERS[preds["CNN"][ix]]}')
    axs[1,k].imshow(CTe[ix,:,:,0],cmap='gray');axs[1,k].imshow(heat,cmap='jet',alpha=.45,vmin=0,vmax=1);axs[1,k].set_title(f'Grad-CAM {LETTERS[k]}' + (' (nulo)' if maximum==0 else ''))
for ax in axs.flat:ax.axis('off')
savefig('gradcam_clases.png')
np.savez_compressed(ROOT/'results/gradcam.npz',maps=np.array(heats),test_filtered_indices=representatives)
pd.DataFrame({'letter':LETTERS,'test_filtered_index':representatives,'csv_row':test.index.to_numpy()[representatives]}).to_csv(ROOT/'results/activation_examples.csv',index=False)

a,b=metrics.loc['MLP'],metrics.loc['CNN']
summary=f"""### Síntesis experimental
- MLP reentrenado: **{a.test_accuracy:.2%}** de accuracy, **{a.f1_macro:.4f}** F1 macro y **{int(a.errors)}** errores.
- CNN: **{b.test_accuracy:.2%}** de accuracy, **{b.f1_macro:.4f}** F1 macro y **{int(b.errors)}** errores.
- Diferencia de accuracy: **{100*(b.test_accuracy-a.test_accuracy):+.2f} puntos porcentuales**.
- Parámetros: **109.126 vs 56.454**. El costo convolucional depende también de operaciones y hardware, no solo de pesos.
- Tiempos de fit: **{a.fit_seconds:.2f} s vs {b.fit_seconds:.2f} s**, misma CPU y protocolo.
- Brecha train-test: **{a.train_test_gap_pp:.2f} pp vs {b.train_test_gap_pp:.2f} pp**.
- Evidencia histórica E1: 87,03% y F1 macro 0,8694. Debe distinguirse del MLP reentrenado.
"""
display(Markdown(summary));(ROOT/'results/summary.md').write_text(summary)
# Verificación de serialización y salida tras volver a cargar los modelos.
for name,x in [('MLP',Xt),('CNN',CTe)]:
    restored=tf.keras.models.load_model(ROOT/'models'/f'{name.lower()}.keras')
    np.testing.assert_allclose(restored(x[:8],training=False).numpy(),probs[name][:8],atol=1e-5)
print('Verificación correcta: modelos guardados, salidas equivalentes y seis probabilidades por imagen.')