wget -r --no-parent http://pathplanning.ru/public/ECMR-2019/engines/ --reject "index.html*"
mkdir -p engine
mv ./pathplanning.ru/public/ECMR-2019/engines/* ./engine
rm -rf pathplanning.ru
