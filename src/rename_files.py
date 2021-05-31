from pathlib import Path

def convert_all_files_in_dir(direct, type_file=False):
    """Sorted all files and rename to base look: className_imgNumber"""
    dir_name = Path(direct).name
    dir_list = sorted(list(Path(direct).rglob('*')))
    print(f'dir {dir_name}')
    iter = 0
    for elem in dir_list:
        path = Path(elem)
        if path.is_file():
            extension = path.suffix
            new_name = dir_name +  '_' + str(iter)
            if type_file and extension != '.jpg':
                if extension == '.jpeg':
                    extension = '.jpg'
                else:
                    raise TypeError(f'{path} has got bad extension. \
                                    Extension must be .jpeg or .jpg')
            new_name += extension
            path.rename(Path(direct, new_name))
            iter += 1

if __name__ == '__main__':
    dir_path = '/home/klenlen/Projects/Primary sorting of SHW/\
                        Garbage classification/TRAIN/recycle'
    convert_all_files_in_dir(dir_path)