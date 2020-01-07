
all:
	@echo
	@echo "Ezek közül lehet választani:"
	@echo "  make download-data"
	@echo "  make preprocess-data"
	@echo "  make download-utonevek-data"
	@echo "  make preprocess-utonevek-data"
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

preprocess-utonevek-data:
	( \
    tail -n +2 osszesnoi.txt | recode l2..u8 | sed "s/$$/\t0/" ; \
    tail -n +2 osszesffi.txt | recode l2..u8 | sed "s/$$/\t1/" ; \
  ) > utonevek.data

download-utonevek-data:
	wget http://www.nytud.mta.hu/oszt/nyelvmuvelo/utonevek/osszesnoi.txt
	wget http://www.nytud.mta.hu/oszt/nyelvmuvelo/utonevek/osszesffi.txt

preprocess-enhu-data:
	( \
    cat hu | sed "s/$$/\t0/" ; \
    cat en | sed "s/$$/\t1/" ; \
  ) > enhu.data

# XXX hu-bol esetleg ki lehetne szedni az irasjeleket
download-enhu-data:
	ssh oliphant.nytud.hu 'zcat /store/share/resources/freqs/hnc1.3/wordfreq/hnc-1.3-sum-wordfreq.txt.gz | head -120000 | recode l2..u8 | cut -f 1' | tr 'A-ZŰÁÉÚŐÓÜÖÍ' 'a-zűáéúőóüöí' | sort -u > hu
	cp -p /usr/share/dict/words en

