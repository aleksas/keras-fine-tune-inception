#!/bin/bash

mkdir -p /tmp/data

FILE=/tmp/download/kagglecatsanddogs_3367a.zip
if [ ! -f $FILE ]; then
  wget --directory-prefix=/tmp/download/ https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
  unzip $FILE -d /tmp/download/
fi

mkdir -p /tmp/data/train/cats
mkdir -p /tmp/data/train/dogs
mkdir -p /tmp/data/validation/cats
mkdir -p /tmp/data/validation/dogs

for ((i=0;i<=999;i++));
do
   cp /tmp/download/PetImages/Cat/$i.jpg /tmp/data/train/cats/
   cp /tmp/download/PetImages/Dog/$i.jpg /tmp/data/train/dogs/
done

for ((i=1000;i<=1400;i++));
do
   cp /tmp/download/PetImages/Cat/$i.jpg /tmp/data/validation/cats/
   cp /tmp/download/PetImages/Dog/$i.jpg /tmp/data/validation/dogs/
done

rm -r -f
