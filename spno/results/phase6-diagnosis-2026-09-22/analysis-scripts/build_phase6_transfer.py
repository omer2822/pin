"""Create and fully verify the separate Mac transfer archive; no source writes."""
import hashlib,json,zipfile,time
from pathlib import Path
BASE=Path('C:/Users/omerm/Documents/Codex/2026-09-18/he')
REPO=Path('C:/Users/omerm/PycharmProjects/pin')
BUNDLE=BASE/'work/phase6-git-bundle/spno/results/phase6-diagnosis-2026-09-22'
OUT=BASE/'outputs/phase6-mac-transfer'
OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((BUNDLE/'local-artifacts-manifest.json').read_text())
archive=OUT/'phase6-standalone-artifacts-bd4e108527-K0.zip'
started=time.time()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
    for item in manifest['artifacts']:
        z.write(REPO/item['path'],arcname=item['path'])
print('Archive written; verifying all member hashes.',flush=True)
with zipfile.ZipFile(archive) as z:
    assert set(z.namelist())=={i['path'] for i in manifest['artifacts']}
    for item in manifest['artifacts']:
        digest=hashlib.sha256()
        with z.open(item['path']) as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
        assert digest.hexdigest()==item['sha256'],item['path']
digest=hashlib.sha256()
with archive.open('rb') as f:
    for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
sha=digest.hexdigest()
(OUT/(archive.name+'.sha256')).write_text(sha+'  '+archive.name+'\n',encoding='ascii')
record=dict(filename=archive.name,bytes=archive.stat().st_size,sha256=sha,member_count=len(manifest['artifacts']),
            verified_all_member_sha256=True,contents='36 checkpoints and 26 dataset files under spno/results/phase6-standalone-artifacts/',
            transfer='Copy this ZIP and its .sha256 sidecar from Windows to the Mac. They are not uploaded by this task.',
            compression='ZIP_STORED; uncompressed to avoid extra CPU work')
(BUNDLE/'transfer-archive.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
(OUT/'transfer-archive.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record|{'local_path':str(archive),'seconds':time.time()-started},indent=2),flush=True)
