
all:
	@echo
	@echo "Ezek közül lehet választani:"
	@echo "  make download-data"
	@echo "  make preprocess-data"
	@echo

preprocess-data:
	( \
    lynx -dump -nolist mek.oszk.hu/00500/00597/html/*htm | \
    grep -i "[a-z]" | grep -v "\.gif" | sed "s/$$/\t0/" ; \
    lynx -dump -nolist mek.oszk.hu/01000/01006/html/*htm | \
    grep -i "[a-z]" | grep -v "\.gif" | sed "s/$$/\t1/" \
  ) > data
# szuper-primitív adatfeldolgozás :)
# a kidumpolt html fájlok minden sora "ér"
# Arany kap egy "\t0"-t a végére -- Petőfi pedig egy "\t1"-et :)

download-data:
	wget -r http://mek.oszk.hu/00500/00597/html
	wget -r http://mek.oszk.hu/01000/01006/html

