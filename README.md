```sh
source ~/torch/bin/activate

unzip UKBB-CMR-images.zip -d images
mv images/Traced/SQ/Traced\ 4\ chamber/ images/Traced/SQ/Traced/
mv images/Traced/SQ/Untraced\ Copies/ images/Traced/SQ/Originals/
mv images/Traced/PI/Original\ Untraced/ images/Traced/PI/Originals/

rm -rf output
rm -rf images/blue
rm -rf images/green
rm -rf images/red
rm -rf images/original
rm -rf */.DS_Store

mkdir images/blue
mkdir images/green
mkdir images/red
mkdir images/original

mkdir output

python pre_process.py
```