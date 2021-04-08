#!/bin/zsh

array=($(ls))

echo $array[1]

for conf_name in *; do
  if grep -q  "${conf_name%.yaml}" "latexsource.tex"
  then
    echo $conf_name
    cp $conf_name ../keep_confs

  else
    # code if not found
fi
done