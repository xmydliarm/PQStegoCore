# Perturbed Quantization Steganography (JPEG)

Objektovo orientovaná implementácia steganografického algoritmu <i>Perturbed Quantization</i> v jazyku C++, zameraného na vkladanie tajných správ do JPEG obrázkov a ich extrakciu z nich, technikou odolnou voči štatistickým metódam detekcie.

## Princíp

<p align="left">
  <img src="https://github.com/user-attachments/assets/121f0467-205f-4297-b2a9-4c56c6094ffe" width="560">
</p>

## Funkcie
- Vkladanie/extrahovanie textových správ do/z JPEG súborov
- Výpočet maximálnej kapacity pre konkrétny obrázok
- Automatická detekcia kvality vstupného obrázka
- Dynamické určenie kvality kompresie
- Detekcia neštandardného procesu kompresie
- Podpora farebných aj šedotónových JPEG obrázkov

## Poznámky
- Vstupné obrázky musia byť v JPEG formáte
- Podporované sú výlučne obrázky komprimované štandardným JPEG procesom
- Kvôli kapacite sú podporované výlučne obrázky s kvalitou Q >= 70 
- Pre extrakciu správy je nevyhnutný pôvodný (neupravený) JPEG obrázok

## Použitie

<b>./pq <režim> [parametre]</b>

### Dostupné režimy:
<b>-capacity <obrázok></b> <br>
<b>-encode <vstup> <výstup> "správa"</b> <br>
<b>-decode <originál> <stego></b> <br>

## Ukážkové príklady

### Výpočet kapacity
<b>./pq -capacity image.jpg</b> <br><br>
<b>Výstup:</b> Estimated capacity: 27607 bits

### Vloženie správy
<b>./pq -encode original.jpg stego.jpg "Tajná správa"</b> <br><br>
<b>Výstup:</b> Message embedded! (Changes: 62)

### Extrahovanie správy
<b>./pq -decode original.jpg stego.jpg</b> <br><br>
<b>Výstup:</b> Decoded message: Tajná správa

## Systémové požiadavky

### OS
- Linux  
- Windows  
- macOS  

### Závislosti
- C++ kompilátor (napr. GCC, Clang, MSVC)  
- CMake (pre generovanie build systému)  
- libjpeg-turbo (je súčasťou projektu <i>lib/libjpeg-turbo</i>)

Vo všeobecnost je kompatibilita definová podporou CMake, C++ a knižnice libjpeg-turbo.

## Architektúra

<p align="left">
  <img src="https://github.com/user-attachments/assets/da64337c-b069-499a-b4c0-0629d6ddad45" width="380">
</p>

## Referencie
https://dde.binghamton.edu/download/pq/Fri05pq.pdf <br>
http://www.ws.binghamton.edu/fridrich/Research/dc-si-5.pdf <br>
https://dde.binghamton.edu/download/pq/PQ_matlab.zip <br>
