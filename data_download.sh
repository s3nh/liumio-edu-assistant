
mkdir -p data/raw
for ds in \
  "ajibawa-2023/Children-Stories-Collection" \
  "ajibawa-2023/Maths-Grade-School" \
  "ajibawa-2023/Education-Young-Children"
do
  echo "Downloading $ds ..."
  huggingface-cli download "$ds" \
     --repo-type dataset \
     --local-dir "data/raw/$(echo $ds | cut -d'/' -f2)" \
     --local-dir-use-symlinks False
done
~