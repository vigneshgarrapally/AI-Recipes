# Datumaro

Dataset Management Framework. Dataset reading, writing, conversion in any direction.

## Installation

    pip install datumaro

## Usage Examples

### 1) Merge two coco_instances datasets

    mkdir merged
    cd merged
    datum create
    datum add -f coco source1/
    datum add -f coco source2/
    cd ../
    datum export -p ./merged/ -o merged_coco_outputs/ -f coco_instances  -- --save-images --image-ext='.png'

> **_NOTE:_** Make sure to copy source1/ and source2/ to the merged folder




    
    


