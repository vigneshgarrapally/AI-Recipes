from pathlib import Path

def list_files(path,valid_extensions,recursive=True):
    """
    List all files in a directory given extensions.
    """
    if valid_extensions is None:
        exts_pattern="*.*"
    exts_pattern = "*.*["+"|".join(valid_extensions)+"]"
    if recursive:
        files = list(Path(path).rglob(exts_pattern))
        return files
    else:
        files = list(Path(path).glob(exts_pattern))
        return files


def list_images(path,valid_extensions=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],recursive=True):
    """
    List all images in a directory recursively.
    pass required extensions to valid_extensions parameter to filter the files.
    """
    images_list=list_files(path,valid_extensions,recursive=recursive)
    return images_list

if __name__ == "__main__":
    import argparse
    myparser = argparse.ArgumentParser(description='List all images in a directory.')
    myparser.add_argument('path',metavar='path',type=str,help='Path to the directory of Images.')
    args=myparser.parse_args()
    print(list_images(args.path))