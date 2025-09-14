import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# *--------------------------------------------------------------------------------------------------*#
# * File Name: filesUtils.py
# * Last Modified: 2023-05-31
# * This is the file utils libs to process the file operation like read/write, copy, move, delete, etc.
# *--------------------------------------------------------------------------------------------------*#


def runCmd(cmd, verbose=False):
    """Run the command in the shell

    Args:
        cmd (str): the command you want to run in the shell
        verbose (bool): show the command or not
    """
    if verbose:
        print("cmd: ", cmd)
    res = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    if (res.returncode == 0) and (verbose):
        print(res.stdout)
    elif verbose:
        print(res.stderr)


def getFileSize(filePath, verbose=False):
    fsize = os.path.getsize(filePath)
    fsize = fsize / 1e6
    if verbose:
        print(f"{filePath} size: {fsize} MB")
    return fsize


# *---------------------------------------Get Path/Path Parser--------------------------------------*#
def isExistDir(dir_path):
    """check whether dir_path exists or not
    Args:
        dir_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    return os.path.exists(dir_path)


def DictPath2DictDir(winPaths):
    # TODO: no idea what this function does
    dirRes = {}
    for var, Path in winPaths.items():
        dirRes[var] = os.path.split(Path)[0]
    return dirRes


def parseFNfromP(path):
    """Extract the file name (with ext) from the path"""
    fileName = os.path.split(path)[1]
    return fileName


def parseDirfromP(path):
    """Extract the dir path from the path"""
    dirPath = os.path.split(path)[0]
    return dirPath


def Win2LinuxPath(winPath):
    winPath = winPath.replace("\\", "/")
    linuxPath = "/mnt/" + winPath[0].lower() + winPath[2:]
    return linuxPath


def Linux2WinPath(linuxPath):
    linuxPath = linuxPath.replace("/", "\\")
    winPath = linuxPath[5].upper() + ":" + linuxPath[6:]
    return winPath


def getDirPathsInDir(dir_path):
    """Get all sub-dirs paths in root-dir path"""
    for root, dirs, files in os.walk(dir_path):
        return dirs


def getFilePathsInDir(dir_path, ext=None):
    """Get all file paths in dir_path, if ext is specified, then get all file paths (sorted) with extension name

    Args:
        dir_path (str): get all file paths in this dir_path
        ext (str): if specified, get all file paths with this extension name
    Returns:
        filePaths(list): a list contain all file paths in the dir_path
    """
    if ext != None and ext[0] != ".":
        ext = "." + ext
    filePaths = []
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            file_extName = os.path.splitext(file_name)[-1]
            if (ext == None) or (file_extName == ext):
                filepath = os.path.join(root, file_name)
                filePaths.append(filepath)
    filePaths = sorted(
        filePaths,
        key=lambda x: int(os.path.splitext(os.path.split(x)[-1])[0].split("-")[-1]),
    )
    return filePaths


def getLatestModelPath(model_dir, verbose=False):
    """Get the latest model path from model_dir based on edit time
    (Attention: It would be better that you explicitly input the latest model path. But to make sure
    this method work correctly, you need to turn on verbose to check the whether the file path is correct)

    Args:
        model_dir (str): the dir path which contains all the model files
    Returns:
        model_path(str): the latest model file path
    """
    latest_file_path = None
    latest_time = 0
    last_file_name = None

    for root, dirs, files in os.walk(model_dir):
        for file_name in files:
            _, ext = os.path.splitext(file_name)
            if ext != ".pth":
                continue
            file_path = os.path.join(root, file_name)
            modifyTime = os.path.getmtime(file_path)
            if modifyTime > latest_time:
                last_file_name = file_name
                latest_time = modifyTime
                latest_file_path = file_path
    if last_file_name == None:
        raise ValueError(
            f"getLatestModelPath: Result is {last_file_name}. Can not get the latest model, check your model_dir argument"
        )
    if verbose:
        print(f"getLatestModelPath: load latest file path {last_file_name}")
    return latest_file_path


# *---------------------------------------IO Operation----------------------------------------------*#
def readDat(file_path, toTensor=False):
    "basic & core func"
    dat = np.fromfile(file_path, dtype="<f")
    if toTensor:
        dat = torch.from_numpy(dat)
    return dat


def saveDat(dat, file_path):
    "basic & core func"
    dat.tofile(file_path, format="<f")


def readImg(file_path):
    "basic & core func"
    img = Image.open(file_path)
    img_arr = np.array(img)
    return img_arr


def yaml_loader(file_path):
    settings = yaml.safe_load(open(file_path))
    return settings


def json_loader(file_path):
    content = None
    with open(file_path, "r") as f:
        content = json.load(f)
    return content


# *---------------------------------------Create/Del/Copy Dir----------------------------------------*#


def ensure_dirs(dir_path, verbose=False):
    "ensure dirs exists, if not, create it"
    upperDir = os.path.dirname(dir_path)
    if not os.path.exists(dir_path):
        if not os.path.exists(upperDir):
            ensure_dirs(upperDir, verbose=verbose)
        if verbose:
            print(f"{dir_path} not exists, create the dir")
        os.mkdir(dir_path)
    else:
        if verbose:
            print(f"{dir_path} exists, no need to create the dir")


# *---------------------------------------Create/Del/Copy Files----------------------------------------*#


def copy_modelSetting(log_base_dir, copy_file_names=None):
    """copy model.py, main_bp.py, etc files to log_base_dir

    Args:
        log_base_dir (str): the log paths you want to save these running files
        copy_file_names (list(str)): a list of file names want to save in log file (main_bp.py etc)
    """
    copy_file_path = []
    if copy_file_names == None:
        dirs_and_files = os.listdir(".")
        for item in dirs_and_files:
            rel_path = os.path.join(".", item)
            if os.path.isfile(rel_path):
                copy_file_path.append(rel_path)
    else:
        for item in copy_file_names:
            copy_file_path.append(os.path.join(".", item))

    for one_path in copy_file_path:
        dst_path = os.path.join(log_base_dir, one_path)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.copyfile(one_path, dst_path)


def delDirsInDir(dir_path):
    """Delete all dirs in that dir

    Args:
        dir_path (str): _description_
    """
    for f in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, f)):
            shutil.rmtree(os.path.join(dir_path, f))


def delFilesInDir(dir_path, ext=None):
    """Delete all files in that dir, if ext is specified, then delete all files with that extension name

    Args:
        dir_path (str): _description_
        ext (str, optional): _description_. Defaults to None.
    """
    if ext != None and ext[0] != ".":
        ext = "." + ext
    for f in os.listdir(dir_path):
        file_extName = os.path.splitext(f)[-1]
        if (ext == None) or (file_extName == ext):
            os.remove(os.path.join(dir_path, f))


def volumeRenderDeleteBlankImg(dirPath, timeSteps):
    """Delete all dummy files generated by inhouse volume rendering code

    Args:
        dirPath (str): the dir path for where you save volume rendering image result
        timeSteps (int): total timestep of the rendered object
    """
    paths = sorted(getFilePathsInDir(dirPath))
    for i in range(timeSteps, len(paths)):
        os.remove(paths[i])


def TypeA2TypeB(dir_path, saved_dir_path, typeA=".dat", typeB=".raw", verbose=False):
    """Copy and save the file in dir_path with ext name typeA to saved_dir_path and rename its ext name to typeB

    Args:
        dir_path (_type_): the oringinal data dir
        saved_dir_path (_type_): the new dir to save the data (if not exists, automatically create it)
        typeA (str, optional): specify the ext name in data dir_path. Defaults to '.dat'.
        typeB (str, optional): specify the ext name in data saved_dir_path. Defaults to '.raw'.
    """
    ensure_dirs(saved_dir_path)  # create the saved dir path if the path do not exists
    delFilesInDir(saved_dir_path)
    if verbose:
        print(f"Transform {typeA} to {typeB}")
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            if os.path.splitext(file_name)[-1] == typeA:  # to avoid other types of files
                dat_file_path = os.path.join(root, file_name)
                raw_file_path = os.path.join(saved_dir_path, os.path.splitext(file_name)[0] + typeB)
                shutil.copyfile(dat_file_path, raw_file_path)


def RenameFileExtInDir(dir_path, original_ext=".dat", rename_ext=".iw", verbose=False):
    """Batch operation for changing the file ext name in a dir

    Args:
        dir_path (str): the dir path you want to change your file ext name
        original_ext (str): original ext name
        rename_ext (str): new ext name
        verbose (bool): show the progress bar or not
    """
    for root, dirs, files in os.walk(dir_path):
        for file_name in tqdm(files, disable=(not verbose), desc="Rename file ext"):
            if os.path.splitext(file_name)[-1] == original_ext:  # just to avoid conner case
                dat_file = os.path.join(root, file_name)
                raw_file = os.path.join(root, os.path.splitext(file_name)[0] + rename_ext)
                shutil.move(dat_file, raw_file)


# *---------------------------------------Others(Archived)--------------------------------------------*#


def ppm2png(dirPath):
    """Translate PPM to PNG in-place way

    Args:
        dirPath (str): the ppm dir path
    """
    for root, dirs, files in os.walk(dirPath):
        for file_name in files:
            file_nameExt = os.path.splitext(file_name)[-1]
            file_namePre = os.path.splitext(file_name)[0]
            if file_nameExt == ".ppm":
                filePath = os.path.join(root, file_name)
                newFilePath = os.path.join(root, file_namePre + ".png")
                im = Image.open(filePath)
                im.save(newFilePath)
                os.remove(filePath)


def getYMD():
    """return current time with format YearMonthDay (e.g. 20220928)

    Returns:
        YMDString(str): the format string YearMonthDay
    """
    timeArray = time.localtime(time.time())
    YMDString = time.strftime(r"%Y%m%d", timeArray)
    return YMDString


if __name__ == "__main__":
    getDiffImgInDir(r"F:\Yunhao\H+")
