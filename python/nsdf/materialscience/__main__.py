import os
import sys
import argparse
import logging
import time
import datetime
import shlex
import subprocess
import tempfile
import shutil
from multiprocessing.pool import ThreadPool
from pprint import pprint

import prefect
from prefect import Flow, Parameter, Task, task, unmapped

# you need to install the code from https://github.com/nsdf-fabric/nsdf-software-stack
from nsdf.kernel import logger, LoadYaml, rmfile, GetPackageFilename, RunCommand, SetupLogger, LoadYaml
from nsdf.s3 import S3
from nsdf.distributed import NSDFDaskCluster


# ////////////////////////////////////////////////////////////////////////
def ParseRangeFromString(value):
    ret = value
    if isinstance(ret, str):
        ret = [int(it) for it in ret.split() if it]
    assert(len(ret) == 3)
    return ret


# ////////////////////////////////////////////////////////////////////////////
def FixTensorFlowProblem():
    """
    resolve the problem that `keras` (used by the segmentation) is taking all the GPU memory from all devices
    not leaving GPU memory for the reconstruction
    https://stackoverflow.com/questions/65723891/how-to-free-tf-keras-memory-in-python-after-a-model-has-been-deleted-while-othe
    """

    import tensorflow as tf

    # print statistics of all GPUs (THIS is slow on FLuidStack?)
    physical = tf.config.list_physical_devices('GPU')
    logger.info(f"FixTensorFlowProblem found physical gpu: {physical}")

    # please find a way so I can see only one gpu (with CUDA_VISIBLE_DEVICES)
    if len(physical) != 1:
        error_msg = f"FixTensorFlowProblem ERROR: I was expecting to found only one GPU and found {len(physical)}"
        logger.error(error_msg)
        raise Exception(error_msg)

    for gpu in physical:
        gpu_details = tf.config.experimental.get_device_details(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)

    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    logger.info(
        f"FixTensorFlowProblem CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} done")

# ////////////////////////////////////////////////////////////////////////
def Preprocess(
        hdf5_url=None,
        tot_slices=0,
        rotation_center=None,
        disable_reconstruction=False,
        reconstruction_version=2,
        disable_segmentation=False,
        summarize=False,
        slice_range=[0, -1, 1],
        slab_size=5,
        write_tiff_digit=5,
        uploader_num_connections=8
):

    assert tot_slices
    assert disable_reconstruction or rotation_center

    T1 = time.time()

    loc = os.environ["LOCAL"]
    rem = os.environ["REMOTE"]

    assert hdf5_url.startswith("s3://")
    rem_hdf5 = hdf5_url

    rotation_center = float(rotation_center)

    s3 = S3(num_connections=uploader_num_connections)
    bucket, key, qs = S3.parseUrl(rem_hdf5)
    key = os.path.basename(key)

    loc_hdf5 = f"{loc}/hdf5/{key}"

    r_prefix = f"workflow/{key}/r/tif"
    s_prefix = f"workflow/{key}/s/tif"

    # support rehentrant i.e. if I already have the tiff files avoid to reproduce them
    s3_r = [obj for obj in s3.listObjects(f"{rem}/{r_prefix}", verbose=False) if not obj['url'].endswith("/")]
    s3_s = [obj for obj in s3.listObjects(f"{rem}/{s_prefix}", verbose=False) if not obj['url'].endswith("/")]

    if summarize:
        rnum = len(s3_r)
        snum = len(s3_s)
        rsize = sum([int(it['Size']) for it in s3_r])
        ssize = sum([int(it['Size']) for it in s3_s])
        return (rem_hdf5, tot_slices, rnum, rsize, snum, ssize)

    s3_r = [obj['url'] for obj in s3_r]
    s3_s = [obj['url'] for obj in s3_s]

    logger.info(
        f" Preprocess hdf5({rem_hdf5}) rotation-center({rotation_center}) slice-range({slice_range}) recontructions({len(s3_r)}/{tot_slices}) segmentations({len(s3_s)}/{tot_slices}) ...")

    uploader = s3.createUploader()

    white_universal = None
    trained_model = None

    # do slice by slice
    for I in range(slice_range[0], slice_range[1] if slice_range[1] > 0 else tot_slices, slice_range[2]):

        slice_uid = str(I).rjust(write_tiff_digit, '0')
        loc_rec = f"{loc}/{r_prefix}/i_{slice_uid}.tiff"
        rem_rec = f"{rem}/{r_prefix}/i_{slice_uid}.tiff"
        loc_seg = f"{loc}/{s_prefix}/i_{slice_uid}.tiff"
        rem_seg = f"{rem}/{s_prefix}/i_{slice_uid}.tiff"

        # logger.info(f"Doing slice {I} range {slice_range} rem_hdf5 {rem_hdf5} rotation_center {rotation_center} ...")
        if not disable_reconstruction:

            if rem_rec in s3_r:
                # logger.info(f"Reconstruction {rem_rec} already exists, skipping")
                pass

            else:
                # remmember the recontruction happens in chunks
                I1 = (I // slab_size) * slab_size
                I2 = min(I1+slab_size, tot_slices)

                if not os.path.isfile(loc_rec):
                    try:
                        s3.downloadObject(rem_hdf5, loc_hdf5)

                        t1 = time.time()

                        # internally it will add a suffix SUCH AS `_<num>.tiff``
                        write_tiff_filename = f"{loc}/{r_prefix}/i"

                        # reconstruct versioning here
                        if reconstruction_version == 1:
                            from nsdf.material_science_workflow.reconstruct_v1 import reconstruct_image
                            vmin, vmax = reconstruct_image(
                                loc_hdf5, write_tiff_filename, I1, I2, rotation_center, write_tiff_digit=write_tiff_digit)

                        elif reconstruction_version == 2:
                            from nsdf.material_science_workflow.reconstruct_v2 import reconstruct_image

                            # generate the white  for reconstruction
                            if white_universal is None:
                                t1 = time.time()
                                logger.info(f"Loading white...")
                                white_key = "fly_scan_id_112509.h5"
                                s3.downloadObject(f"s3://Pania_2021Q3_in_situ_data/hdf5/{white_key}", f"{loc}/hdf5/{white_key}")
                                import h5py
                                Data_Universal = h5py.File(
                                    f"{loc}/hdf5/{white_key}", 'r')
                                white_universal = Data_Universal['img_bkg']
                                del Data_Universal
                                sec = time.time()-t1
                                logger.info(f"Loaded white in {sec} seconds")

                            vmin, vmax = reconstruct_image(
                                loc_hdf5, write_tiff_filename, I1, I2, rotation_center, white_universal, write_tiff_digit=write_tiff_digit)

                        else:
                            raise Exception(
                                f"unsupported version {reconstruction_version}")

                        logger.info(
                            f" Reconstruction {rem_hdf5} {I1}/{I2}vmin {vmin} vmax {vmax} loc_rec({loc_rec}) reconstruction_version({reconstruction_version})  done in {time.time()-t1} seconds")
                        assert os.path.isfile(loc_rec)

                    except Exception as ex:
                        # I want to see the error on the master terminal
                        logger.error(ex, exc_info=True)

                if os.path.isfile(loc_rec):
                    s3.uploadObject(loc_rec, rem_rec)
            #		uploader.put(loc_rec, rem_rec)  ###possible bug on thread concurrency issues, comment out to use single upload

        if not disable_segmentation:

            if rem_seg in s3_s:
                # logger.info(f"Segmentation {rem_rec} already exists, skipping")
                pass

            else:
                if not os.path.isfile(loc_seg):

                    if trained_model is None:
                        t1 = time.time()
                        from nsdf.material_science_workflow.segment_v1 import load_new_model
                        logger.info(f" Loading trained model...")
                        trained_model = load_new_model(GetPackageFilename(
                            "material_science_workflow/resources/seg_msd_50_2_ep100"))
                        logger.info(
                            f"Loaded trained model in {time.time()-t1} seconds")

                    try:
                        s3.downloadObject(rem_rec, loc_rec)
                        t1 = time.time()
                        from nsdf.material_science_workflow.segment_v1 import process_whole_image, load_new_model
                        assert os.path.basename(
                            loc_rec) == os.path.basename(loc_seg)
                        bname = os.path.basename(loc_rec)
                        process_whole_image(os.path.dirname(
                            loc_rec),  os.path.dirname(loc_seg), bname, trained_model)
                        assert os.path.isfile(loc_seg)
                        logger.info(
                            f" Segmentation {rem_hdf5} {I}/{tot_slices} {loc_seg} done in {time.time()-t1} seconds")

                    except Exception as ex:
                        # I want to see the error on the master terminal
                        logger.error(ex, exc_info=True)

                if os.path.isfile(loc_seg):
                    s3.uploadObject(loc_seg, rem_seg)
                    #uploader.put(loc_seg, rem_seg)

    # wait for all files
    # try:
         #       uploader.waitAndExit()

    # except KeyboardInterrupt:
         #       print("no traceback")
    # uploader.waitAndExit() ###possible bug on thread concurrency issues, comment out to use single upload

    # clean up
    if True:
        rmfile(loc_hdf5)
        shutil.rmtree(f"{loc}/{r_prefix}", ignore_errors=True)
        shutil.rmtree(f"{loc}/{s_prefix}", ignore_errors=True)

    # free GPU memory
    if trained_model:
        del trained_model

    logger.info(f"Preprocess {rem_hdf5} done in {time.time()-T1} seconds")
    return True


# ////////////////////////////////////////////////////////////////////////
def CallPreprocess(d):
    return Preprocess(**d)

# ////////////////////////////////////////////////////////////////////////


def CollectPreprocessStats(stats, key, tot_slices, rnum, rsize, snum, ssize):
    rperc, r_sec_per_slice, eta_r = int(100.0*rnum/tot_slices), 0, 0
    sperc, s_sec_per_slice, eta_s = int(100.0*snum/tot_slices), 0, 0
    MiB = 1024*1024

    if key in stats:
        sec = time.time()-stats[key]["t1"]
        r_diff = rnum-stats[key]["rnum"]
        r_sec_per_slice = sec/r_diff if r_diff else 0
        s_diff = snum-stats[key]["snum"]
        s_sec_per_slice = sec/s_diff if s_diff else 0
        eta_r = int((tot_slices-rnum)*r_sec_per_slice/60)
        eta_s = int((tot_slices-snum)*s_sec_per_slice/60)

    stats[key] = {"t1": time.time(), "rnum": rnum,  "snum": snum}


# ////////////////////////////////////////////////////////////////////////////
@task
def PreprocessTask(d):
    return CallPreprocess(d)


def PreprocessMain(workflow):

    env = workflow["env"]
    files = workflow["files"]
    tot_slices = files[0]["tot-slices"]  # assume they are alll the same
    summarize = "--summarize" in sys.argv

    # file range
    if True:
        file_range = workflow["file-range"]
        if isinstance(file_range, str):
            file_range = ParseRangeFromString(file_range)
        if file_range[1] < 0:
            file_range[1] = len(files)
        files = [files[I] for I in range(*file_range)]

    # slice range
    if True:
        slice_range = workflow["slice-range"]
        if isinstance(slice_range, str):
            slice_range = ParseRangeFromString(slice_range)

        if slice_range[1] < 0:
            slice_range[1] = tot_slices

    ARGS = [{
        "hdf5_url": file["url"],
        "rotation_center": file["rotation-center"],
            "reconstruction_version": file["reconstruction-version"],
            "tot_slices": int(file["tot-slices"]),
            "disable_reconstruction": workflow["disable-reconstruction"],
            "disable_segmentation": workflow["disable-segmentation"],
            "slice_range": slice_range,
            "summarize": summarize,
            }
            for file in files]

    if summarize:
        stats = {}
        p = ThreadPool(128)
        t1 = time.time()
        result = p.map(CallPreprocess, ARGS)
        TOT_SLICES, RNUM, RSIZE, SNUM, SSIZE = [0]*5
        for (rem_hdf5, tot_slices, rnum, rsize, snum, ssize) in result:
            CollectPreprocessStats(stats, os.path.basename(
                rem_hdf5), tot_slices, rnum, rsize, snum, ssize)
            TOT_SLICES, RNUM, SNUM, RSIZE, SSIZE = TOT_SLICES + \
                tot_slices, RNUM+rnum, SNUM+snum, RSIZE+rsize, SSIZE+ssize
        CollectPreprocessStats(stats, "TOTAL", TOT_SLICES,
                               RNUM, RSIZE, SNUM, SSIZE)
        logger.info("")
        sys.exit(0)

    if "dask" in workflow and bool(workflow["dask"].get("enabled", False)):

        inventory = workflow["dask"]["inventory"]
        group = workflow["dask"]["group"]
        num_process_per_host = workflow["dask"]["num-process-per-host"]
        worker_local_dir = workflow["dask"]["worker-local-dir"]

        """
		PYTHONOPTIMIZE=1 
			disable assert on workers because of this problem: `AssertionError: daemonic processes are not allowed to have children`
			this happens with Preprocessing GPU on Dask with tomopy package
			see https://stackoverflow.com/questions/46188531/what-does-pythonoptimize-do-in-the-python-interpreter/57493983#57493983
			IMPORTANT TO REMEMBER: it will disable all asserts as well
		"""
        env["PYTHONOPTIMIZE"] = "1"

        cluster = NSDFDaskCluster({
            "inventory": inventory,
            "group": group,
            "num-process-per-host": num_process_per_host,
            "worker-local-dir": worker_local_dir,
            "need-cuda": True,
            "env": env
        })

        def SetupWorkerEnv():
            import os
            assert "AWS_ACCESS_KEY_ID" in os.environ

            # check that I don't see multiple GPUs
            devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            assert devices is None or len(devices.split(",")) == 1
            logger.info(f"DASK worker env READY")

        with cluster.connect() as client:
            client.run(FixTensorFlowProblem)
            client.run(SetupWorkerEnv)
            tasks = []
            with Flow("nsdf-flow") as flow:
                for args in ARGS:
                    tasks.append(PreprocessTask(args))
            state = cluster.execute(flow, tasks)

        cluster.close()

    else:
        FixTensorFlowProblem()
        for args in ARGS:
            CallPreprocess(args)


def CheckObjectExist(it):
    s3 = S3(num_connections=1)
    return (it["creates"], s3.existObject(it["creates"]))


@task
def ConvertImageStackTask(d):
    from nsdf.convert import ConvertImageStack
    return ConvertImageStack(**d)


def ConvertImageStackMain(workflow):

    env = workflow["env"]
    files = workflow.pop("files")

    summarize = "--summarize" in sys.argv

    # file range
    if True:
        file_range = workflow["file-range"]

        if isinstance(file_range, str):
            file_range = ParseRangeFromString(file_range)

        if file_range[1] < 0:
            file_range[1] = len(files)

        files = [files[I] for I in range(*file_range)]

    rem = os.environ["REMOTE"]
    loc = os.environ["LOCAL"]

    disable_reconstruction = workflow["disable-reconstruction"]
    disable_segmentation = workflow["disable-segmentation"]
    arcos = [it for it in workflow["convert"]["arco"].split() if it]
    keep_local_image_stack = bool(
        workflow["convert"]["keep-local-image-stack"])
    keep_local_idx = bool(workflow["convert"]["keep-local-idx"])

    ARGS = []
    for file in files:

        bucket, key, qs = S3.parseUrl(file["url"])
        key = os.path.basename(key)

        whats = []
        if not disable_reconstruction:
            whats.append("r")
        if not disable_segmentation:
            whats.append("s")

        for arco in arcos:
            for what in whats:
                ARGS.append({
                    "creates":  f"{rem}/workflow/{key}/{what}/idx/{arco}/.done",
                    "src": {
                                "remote": f"{rem}/workflow/{key}/{what}/tif/**/*.tiff",
                                "local": f"{loc}/workflow/{key}/{what}/tif/**/*.tiff",
                                "keep-local": keep_local_image_stack,
                    },
                    "dst": {
                        "local": f"{loc}/workflow/{key}/{what}/idx/{arco}/visus.idx",
                        "remote": f"{rem}/workflow/{key}/{what}/idx/{arco}/visus.idx",
                        "keep-local": keep_local_idx,
                        "arco": arco
                    },
                })

    if summarize:
        stats = {}
        p = ThreadPool(128)
        result = p.map(CheckObjectExist, ARGS)
        for it in result:
            print(*it)
        sys.exit(0)

    # run in parallel on dask?
    if "dask" in workflow and bool(workflow["dask"].get("enabled", False)):

        inventory = workflow["dask"]["inventory"]
        group = workflow["dask"]["group"]
        num_process_per_host = workflow["dask"]["num-process-per-host"]
        worker_local_dir = workflow["dask"]["worker-local-dir"]

        cluster = NSDFDaskCluster({
            "inventory": inventory,
            "group": group,
            "num-process-per-host": num_process_per_host,
            "worker-local-dir": worker_local_dir,
            "need-cuda": False,
            "env": env
        })

        def SetupWorker():
            import os
            # check I am getting the environment variables
            assert "AWS_ACCESS_KEY_ID" in os.environ
            logger.info(f"DASK worker env READY")

        with cluster.connect() as client:
            client.run(SetupWorker)
            tasks = []
            with Flow("nsdf-flow") as flow:
                for args in ARGS:
                    tasks.append(ConvertImageStackTask(args))
            state = cluster.execute(flow, tasks)

        cluster.close()

    # run serially on the current host
    else:
        for args in ARGS:
            from nsdf.convert import ConvertImageStack
            ConvertImageStack(**args)


# ////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":

    workflow = LoadYaml(os.path.join(
        os.path.dirname(__file__), "workflow.yaml"))

    # enviroment from the workflow
    if True:
        from nsdf.kernel import NormalizeEnv, PrintEnv, SetEnv
        env = NormalizeEnv(workflow["env"])
        if "export-env" in sys.argv:
            PrintEnv(env)
            sys.exit(0)
        SetEnv(env)

    # setup logging
    if True:
        import logging
        os.makedirs("/tmp/nsdf", exist_ok=True)
        SetupLogger(logger, level=logging.INFO, handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/tmp/nsdf/nsdf.log")])

    task = workflow["task"]
    if task == "preprocess":
        PreprocessMain(workflow)

    elif task == "convert":
        ConvertImageStackMain(workflow)

    else:
        raise Exception("not supported")
