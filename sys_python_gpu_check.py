import torch, platform, os
print('python', platform.python_version())
print('torch', getattr(torch,'__version__','not installed'))
print('cuda_available', torch.cuda.is_available() if hasattr(torch,'cuda') else False)
print('cuda_version', getattr(getattr(torch,'version',None),'cuda',None))
print('num_devices', torch.cuda.device_count() if hasattr(torch,'cuda') else 0)
print('env CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES'))
