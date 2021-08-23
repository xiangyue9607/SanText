
mkdir ./data
cd ./data

wget -O SST-2.zip https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
wget -O QNLI.zip https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip

wget -O glove.840B.300d.zip https://nlp.stanford.edu/data/glove.840B.300d.zip

wget -O wikitext-2.zip https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip

unzip SST-2.zip
unzip QNLI.zip
unzip glove.840B.300d.zip
unzip wikitext-2.zip

