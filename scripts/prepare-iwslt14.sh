#!/usr/bin/env bash
# Adapted from https://github.com/pytorch/fairseq/blob/master/
#                    examples/translation/prepare-iwslt14.sh

if [ ! -d mosesdecoder ]; then 
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git
fi


SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPE_TOKENS=14000

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
prep=iwslt
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep


URL14="https://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz"
GZ14=de-en.tgz
URL15="https://wit3.fbk.eu/archive/2015-01//texts/de/en/de-en.tgz"
GZ15=iwslt15.tgz

cd $orig

echo "Downloading data from ${URL15}..."
wget -O $GZ15  $URL15
if [ -f $GZ15 ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ15 
echo "Copying tst2013 from IWSLT15"
mv $lang/IWSLT15.TED.tst2013.de-en.en.xml .
mv $lang/IWSLT15.TED.tst2013.de-en.de.xml .
rm -rf $lang

echo "Downloading data from ${URL14}..."
wget $URL14
if [ -f $GZ14 ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi
tar zxvf $GZ14
mv IWSLT15.TED.tst2013.de-en.en.xml $lang/IWSLT14.TED.tst2013.de-en.en.xml
mv IWSLT15.TED.tst2013.de-en.de.xml $lang/IWSLT14.TED.tst2013.de-en.de.xml

rm $GZ14 $GZ15
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done


echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/train.$l

    cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
        $tmp/IWSLT14.TED.tst2010.de-en.$l \
        $tmp/IWSLT14.TED.tst2011.de-en.$l \
        $tmp/IWSLT14.TED.tst2012.de-en.$l \
        $tmp/IWSLT14.TED.tst2013.de-en.$l \
        > $tmp/test.$l
done

TRAIN=$tmp/train.en-de
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "Learning BPE on ${TRAIN}..."

subword-nmt learn-bpe -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "Applying BPE to ${f}..."
        subword-nmt apply-bpe -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done

echo "Moving $prep to the data folder"
mkdir -p ../data/
mv $prep ../data/

