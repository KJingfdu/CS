#!/bin/bash

# Define the file URLs and MD5 checksums
files=(
    "http://example.com/ILSVRC2012_img_train.tar.gz"
    "http://example.com/ILSVRC2012_img_val.tar.gz"
    # Add more files as necessary
)

md5sums=(
    "1d675b47d978889d74fa0da5fadfb00e"
    "29b22e2961454d5413ddabcf34fc5622"
    # Add the corresponding MD5s for the files
)

# Define the download directory
download_dir="./data/ImageNet"

mkdir -p $download_dir
cd $download_dir

# Download files and check their MD5
for i in "${!files[@]}"; do
    file=$(basename "${files[i]}")
    wget -c "${files[i]}" -O "$file"

    # Check MD5
    echo "${md5sums[i]}  $file" | md5sum -c -
    if [ $? -ne 0 ]; then
        echo "MD5 mismatch for $file, re-downloading..."
        rm "$file"
        wget -c "${files[i]}" -O "$file"
    fi
done
