#!/usr/bin/env bash

if [ ! -d mosesdecoder ]; then 
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git
fi


SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPE_TOKENS=10000

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=vi
lang=en-vi
prep=envi_word
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep


URL="https://wit3.fbk.eu/archive/2015-01//texts/en/vi/en-vi.tgz"
GZ="en-vi.tgz"

cd $orig

echo "Downloading data from ${URL}..."
wget $URL
if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    grep -v '<speaker>' | \
    grep -v '<reviewer>' | \
    grep -v '<translator>' | \
    grep -v '<\/reviewer>' | \
    grep -v '<\/translator>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done

perl $CLEAN -ratio 15 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 1000
#for l in $src; do
    #perl $LC < $tmp/train.tags.$lang.tok.$l > $tmp/train.tags.$lang.$l
#done

cp $tmp/train.tags.$lang.clean.$tgt  $tmp/train.tags.$lang.$tgt
cp $tmp/train.tags.$lang.clean.$src  $tmp/train.tags.$lang.$src


echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT15.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l > $f
    echo ""
    done
done


echo "creating train, valid, test..."
for l in $src $tgt; do
    cat $tmp/train.tags.en-vi.$l > $tmp/train.$l
    cat $tmp/IWSLT15.TED.tst2013.en-vi.$l > $tmp/test.$l
    cat $tmp/IWSLT15.TED.tst2012.en-vi.$l > $tmp/valid.$l
done

#BPE_CODE=$prep/code

#for l in $src $tgt; do
    #echo "Learning BPE on ${l}..."
    #subword-nmt learn-bpe -s $BPE_TOKENS < $tmp/train.$l > $BPE_CODE.$l
    #for f in train.$l valid.$l test.$l; do
        #echo "Applying BPE to ${f}..."
        #subword-nmt apply-bpe -c $BPE_CODE.$l < $tmp/$f > $prep/$f
    #done

#done
for l in $src $tgt; do
    for f in train.$l valid.$l test.$l; do
        mv $tmp/$f  $prep/$f
    done
done

echo "Moving $prep to the data folder"
mkdir -p ../data/
mv $prep ../data/

