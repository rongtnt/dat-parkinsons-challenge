import glob, os, sys, nibabel as nib, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import gaussian_filter
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
lab = pd.read_csv(f'{D}/train_labels.csv').set_index('uid')['is_pathologic']
def meta(f):
    uid = os.path.basename(f)[:-7]
    im = nib.load(f); d = np.asarray(im.dataobj).astype(np.float32)
    z = im.header.get_zooms()
    return dict(uid=uid, y=lab.get(uid, np.nan), sx=round(float(z[0]),3), sy=round(float(z[1]),3), sz=round(float(z[2]),3),
        nx=im.shape[0], ny=im.shape[1], nz=im.shape[2], dtype=str(im.get_data_dtype()), vmax=float(d.max()), vmean=float(d.mean()),
        p99=float(np.percentile(d,99)), nz_frac=float((d>0).mean()), orient=''.join(nib.aff2axcodes(im.affine)),
        ox=float(im.affine[0,3]), oy=float(im.affine[1,3]), oz=float(im.affine[2,3]), scl=float(im.header['scl_slope']) if im.header['scl_slope'] else 1.0)
if __name__ == '__main__':
    fs = sorted(glob.glob(f'{D}/niftis/*.nii.gz'))
    with ProcessPoolExecutor(16) as ex: rows = list(ex.map(meta, fs, chunksize=8))
    df = pd.DataFrame(rows); df.to_csv(f'{D}/meta_train.csv', index=False)
    df['fmt'] = df['sx'].astype(str)+'|'+df['nx'].astype(str)+'x'+df['ny'].astype(str)+'x'+df['nz'].astype(str)
    g = df.groupby('fmt').agg(n=('uid','size'), pos=('y','mean'), vmax_med=('vmax','median'), p99_med=('p99','median'), nzf=('nz_frac','median'), dtypes=('dtype', lambda s: ','.join(sorted(set(s)))), nz_min=('nz','min'), nz_max=('nz','max')).sort_values('n', ascending=False)
    pd.set_option('display.width', 250); print(g.to_string()); print('orient', df['orient'].value_counts().to_dict()); print('missing labels', df['y'].isna().sum(), 'n', len(df))
    print('fine sx groups', df.groupby('sx').size().to_dict())
