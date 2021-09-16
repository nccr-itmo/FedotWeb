#!/bin/bash
if [[ $(ps --no-headers -o comm 1) == "systemd" ]]
then
	sudo systemctl start mongod
	sudo systemctl status mongod
else
	sudo service mongod start
	sudo service mongod status
fi
sudo printf 'MONGO_CONN_STRING=mongodb://127.0.0.1:27017/test_db?compressors=disabled&gssapiServiceName=mongodb' > mongo_conn_string.env
mongo test_db
