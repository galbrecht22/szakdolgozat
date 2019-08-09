# SZAKDOLGOZAT
Anonim(nak vélt) adathalmazok személyes jellegének tesztelése, különböző algoritmusokkal
 - kld: KL-Divergencia
 - emd: Earth Mover's Distance
 - wmd: Word Mover's Distance

Futtatás menete:
- src/properties.py fájlban kell megadni a vizsgálandó adathalmazt (CABS - lokációs/BMS - online kereskedelmi/MSNBC - webböngészési), illetve a tesztelendő rekordok számát
- $ python src/preprocess.py: adathalmazban rekordok előfeldolgozása
- $ python src/methods/kld.py/emd.py/wmd.py
- *raw fájlokban található a futtatás eredménye
