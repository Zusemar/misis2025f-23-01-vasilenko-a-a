Сборка

На macOS с установленным OpenCV в системе:

```bash
cmake --build /Users/alex/Documents/misis2025f-23-01-vasilenko-a-a/build --config Release -j
```

Исполняемые файлы появляются в `bin.rel/` (для конфигурации Release) либо в соответствующих папках `bin.*` (см. корневой `CMakeLists.txt`).

`task01-01` — проверка соответствия имён файлов формату растра:
`task01-02` - 

```bash
/Users/alex/Documents/misis2025f-23-01-vasilenko-a-a/bin.rel/task01-01 ./bin.rel/task01-01 prj.lab/lab01/testdata/task01.lst
```
Вывод — по одной строке на файл:

```
/abs/path/img_1.png	good
/abs/path/img_2.tiff	bad, should be 0768x0432.3.uint08
```
