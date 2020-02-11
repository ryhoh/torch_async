# torch_async
他モデル適用　検証用ブランチ

## Usage
```
    $ cd Docker

    $ mv batch.sh.sample batch.sh
    $ mv container.yml.sample container.yml

    $ rm nohup.out

    $ nohup sudo docker-compose -f container.yml run --rm -T --name `uuidgen` batch Docker/batch.sh > nohup.log &
```
