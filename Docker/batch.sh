#!/bin/bash

message() {
curl -X POST \
-H 'Authorization: Bearer access_token' \
-F "message=pc_nameで$1" https://notify-api.line.me/api/notify
}

while true
do
        # ファイルに内容が書かれている場合
        if [ -s Docker/Tasks.txt ]; then
                # 1行目を読み込む
                command=`head -n 1 Docker/Tasks.txt`
                # 1行目を削除する
                sed -i -e '1d' Docker/Tasks.txt
        else
                break
        fi
        # コマンド実行
        python3 $command
        # 終了ステータスをチェック
        if [ $? -ne 0 ]; then
                echo $command >> Docker/Failed.txt && message エラー
        else
                echo $command >> Docker/Success.txt
        fi
done

message すべての学習が終了
