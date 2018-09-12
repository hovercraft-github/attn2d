import subprocess
import os
import logging


def set_env(jobname, manual_gpu_id):
    """
    Setup Cuda visible devices
    """
    logger = logging.getLogger(jobname)
    # setup gpu
    try:
        gpu_id = subprocess.check_output(
            'gpu_getIDs.sh', shell=True).decode('UTF-8')
        gpu_ids = gpu_id.split()
        num_gpus = 1
        if len(gpu_ids) > 1:
            gpu_id = ",".join(gpu_ids)
            num_gpus = len(gpu_ids)
        else:
            gpu_id = str(gpu_ids[0])
        logger.warning('Assigned GPUs: %s', gpu_id)
        assert type(gpu_id) == str
    except:
        gpu_id = manual_gpu_id
        num_gpus = len(gpu_id.split(','))
        logger.warning('Selected GPUs: %s', gpu_id)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    return num_gpus
