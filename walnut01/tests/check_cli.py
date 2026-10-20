"""Check inference with a renamed TIFF, no annotation arguments, and bad input."""
from pathlib import Path
import json
import subprocess
import tempfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
binary=ROOT/'build/walnut_segment'
with tempfile.TemporaryDirectory(prefix='walnut-inference-check-') as temporary:
    tmp=Path(temporary)
    image=tmp/'arbitrary_unseen_filename.tif'
    image.write_bytes((ROOT/'data/images/walnut01.0130.tif').read_bytes())
    command=[str(binary),str(image),str(tmp/'output'),'--config',str(ROOT/'configs/recommended.json')]
    subprocess.run(command,check=True,capture_output=True)
    observed=np.asarray(Image.open(tmp/'output/masks/arbitrary_unseen_filename.png'))
    selected=json.loads((ROOT/'results/metrics.json').read_text())['_recommended']['variant']
    expected=np.asarray(Image.open(ROOT/f'results/{selected}/masks/walnut01.0130.png'))
    np.testing.assert_array_equal(observed,expected)
    bad=tmp/'uint8_is_not_float_CT.tif'
    Image.fromarray(np.zeros((30,30),np.uint8)).save(bad)
    process=subprocess.run([str(binary),str(bad),str(tmp/'bad')],capture_output=True,text=True)
    assert process.returncode!=0 and not (tmp/'bad/masks/uint8_is_not_float_CT.png').exists()
print('CLI inference is filename-independent; uint8 input rejected; no annotation arguments used.')
