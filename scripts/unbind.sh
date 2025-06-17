snvme_devices=$(find /sys/bus/pci/drivers/snvme/ -type l -name "????:??:??.?" -printf "%f\n")

if [ -z "$snvme_devices" ]; then
    echo "没有找到绑定在 snvme 驱动上的设备。"
    exit 0
fi

echo "找到以下 snvme 设备将被解绑："
echo "$snvme_devices"

执行解绑
for dev in $snvme_devices; do
    echo -n "$dev" > /sys/bus/pci/drivers/snvme/unbind
    echo "已解绑 $dev"
done

echo "所有 snvme 设备已解绑完成。"