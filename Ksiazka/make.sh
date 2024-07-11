pdflatex -output-directory=build -synctex=1 -interaction=nonstopmode licencjat.tex
cd build/; biber licencjat.bcf; cd ..
pdflatex -output-directory=build -synctex=1 -interaction=nonstopmode licencjat.tex
pdflatex -output-directory=build -synctex=1 -interaction=nonstopmode licencjat.tex
