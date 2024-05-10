import os


def _collect_filenames(datapath):
    print(os.walk(datapath))
    filenames = []
    subfolders = []
    for root, dirs, files in os.walk(datapath):
        print(root)
        for file in files:
            if "_mask" not in file:
                filenames.append(os.path.join(root, file))
                subfolder = os.path.relpath(root, datapath)
                subfolders.append(subfolder)
    return filenames, subfolders


filenames, subfolders = _collect_filenames("Dataset_BUSI_with_GT/")
