import os

def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG']):
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir


def get_list_file_in_dir_and_subdirs(folder, ext=['jpg', 'png', 'JPG', 'PNG']):
    file_names = []
    for path, subdirs, files in os.walk(folder):
        for name in files:
            extension = os.path.splitext(name)[1].replace('.', '')
            if extension in ext:
                file_names.append(os.path.join(path, name).replace(folder, '')[1:])
                # print(os.path.join(path, name).replace(folder,'')[1:])
    return file_names


def get_list_dir_and_subdirs_in_folder(folder):
    list_dir = [x[0].replace(folder, '').lstrip('/') for x in os.walk(folder)]
    return list_dir

