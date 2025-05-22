/**
 * 信息
 */

'use strict';

const time = document.querySelector('#time');
$(document).ready(function () {
    $("#info").fadeTo("slow", 1.0);
    time.onload = setInterval(() => time.innerHTML = new Date().toLocaleString().slice(9, 19));
})

const upCount = document.querySelector('#uploadCount');
const csrftoken = $('meta[name=csrf-token]').attr('content');
var uploadCount = 0,
    uploadList = [];

var uploads = weui.uploader('#uploader', {
    url: '/file_upload',
    auto: false,
    type: 'file',
    fileVal: 'file',
    compress: {
        width: 1600,
        height: 1600,
        quality: .8
    },
    onBeforeQueued: function (files) {
        // `this` 是轮询到的文件, `files` 是所有文件
        if (["image/jpg", "image/jpeg", "image/png"].indexOf(this.type) < 0) {
            weui.alert('请上传jpg或png格式的图片');
            return false;
        }
        if (this.size > 6 * 1024 * 1024) {
            weui.alert('请上传不超过6M的图片');
            return false;
        }
        if (files.length > 1) {
            weui.alert('最多只能上传1张图片，请重新选择');
            return false;
        }
        if (uploadCount + 1 > 1) {
            weui.alert('最多只能上传1张图片');
            return false;
        }
        ++uploadCount;
    },
    onQueued: function () {
        uploadList.push(this);
        upCount.innerHTML = uploadCount;
    },
    onBeforeSend: function (data, headers) {
        $.extend(headers, {
            'X-CSRFToken': csrftoken
        });
        $.extend(headers, {
            'Type': 2
        });
    },
    onProgress: function (procent) {
        $("#upload").hide(100);
        $("#loading-0").show(1000);
    },
    onSuccess: function (ret) {
        if (ret.code == 1) {
            info.ifSameFace(ret.user_id);
        } else if (ret.code == 0) {
            layer.msg(ret.msg);
        } else {
            layer.msg('未知错误')
        }
    },
    onError: function (err) {
        layer.msg('失败');
    }
});

/**
 * 缩略图预览
 */
$("#uploaderFiles").on('click', function (e) {
    var target = e.target;
    while (!target.classList.contains('weui-uploader__file') && target) {
        target = target.parentNode;
    }
    if (!target) return;
    var url = target.getAttribute('style') || '';
    var id = target.getAttribute('data-id');
    if (url) {
        url = url.match(/url\((.*?)\)/)[1].replace(/"/g, '');
    }
    var gallery = weui.gallery(url, {
        onDelete: function onDelete() {
            weui.confirm('确定删除该图片？', function () {
                --uploadCount;
                upCount.innerHTML = uploadCount;
                uploadList.splice(0, 1);
                target.remove();
                gallery.hide();
            });
        }
    });
});

/**
 * 上传按钮监听
 */
$("#upload").on('click', function () {
    weui.confirm('确认上传？', function () {
            if (uploadList.length === 1 && uploadCount === 1) {
                uploadList.forEach(function (file) {
                    file.upload();
                });
            } else {
                layer.msg('请选择一张照片');
            }
        },
        function () {
            console.log('no');
        });
});

/**
 * 获取原始数据按钮
 */
$("#get-data").on('click', function () {//如果点击了get-data元素
    if (Cookie.get('user_id') != undefined) {//检查cookie中是否有user_id的值
        $("#get-data").hide(100);
        $("#loading-1").show(1000);
        layer.msg('处理中');
        info.getOriginData();
    } else {
        layer.msg('请先上传');
    }
});
$("#get-data1").on('click', function () {//如果点击了get-data1元素
    if (Cookie.get('user_id') != undefined) {//检查cookie中是否有user_id的值
        $("#get-data1").hide(100);
        $("#loading-3").show(1000);
        layer.msg('处理中');
        info.getOriginData1();
    } else {
        layer.msg('请先上传');
    }
});
/**
 * 获取加密数据按钮
 */
$("#get-en-data").on('click', function () {
    if (Cookie.get('user_id') != undefined) {
        $("#get-en-data").hide(100);
        $("#loading-2").show(1000);
        layer.msg('处理中');
        info.getEncryptData();
    } else {
    	layer.msg('请先上传');
    }
});
$("#get-en-data1").on('click', function () {
    if (Cookie.get('user_id') != undefined) {
        $("#get-en-data1").hide(100);
        $("#loading-4").show(1000);
        layer.msg('处理中');
        info.getEncryptData1();
    } else {
    	layer.msg('请先上传');
    }
});
$("#cal").on('click', function () {
    $.ajax({
        type: 'POST',
        url: '/cal',
        dataType: 'json',
        headers: {
            'X-CSRFToken': csrftoken
        }, 
        success: function (res) {
            if (res && res.user_id) {
                var user_id = res.user_id;
                $("#cal").hide(100);
                $("#loading-6").show(1000);
                layer.msg('处理中');
                info.calcu(user_id);
            } else {
                layer.msg('无法获取用户ID');
            }
        },
        error: function () {
            layer.msg('请求失败');
        }
    });
});
var info = {
    calcu: function (user_id) {
        $.ajax({
            type: 'POST',
            url: '/face_compare',
            dataType: 'json',
            headers: {
                'X-CSRFToken': csrftoken
            },
            data: {
                user_id: user_id
            },
            success: function (data) {
                if (data.code == 1) {
                    layer.msg(data.msg);
                    $("#calres").html(data.msg + '，判断值为：' + data.data);
                    $("#calres").removeClass('text-danger');
                    $("#calres").addClass('text-success');
                    $("#loading-6").hide();
                    $("#cal").show(500);
                } else if (data.code == 2) {
                    layer.msg(data.msg);
                    $("#calres").html(data.msg + '，判断值为：' + data.data);
                    $("#calres").removeClass('text-success');
                    $("#calres").addClass('text-danger');
                    $("#loading-6").hide(100);
                    $("#cal").show(500);
                } else {
                    layer.msg('错误');
                    $("#loading-6").hide(100);
                    $("#cal").show(500);
                }
            },
            error: function (data) {
            	layer.msg('错误');
            	$("#loading-6").hide(100);
                $("#cal").show(500);
            	console.log(data);
            },
        });
    },
    ifSameFace: function (user_id) {
        $.ajax({
            type: 'POST',
            url: '/face_compare',
            dataType: 'json',
            headers: {
                'X-CSRFToken': csrftoken
            },
            data: {
                user_id: user_id
            },
            success: function (data) {
                if (data.code == 1) {
                    layer.msg(data.msg);
                    $("#tips-0").html(data.msg + '，判断值为：' + data.data);
                    $("#tips-0").removeClass('text-danger');
                    $("#tips-0").addClass('text-success');
                    $("#loading-0").hide();
                    $("#upload").show(500);
                } else if (data.code == 2) {
                    layer.msg(data.msg);
                    $("#tips-0").html(data.msg + '，判断值为：' + data.data);
                    $("#tips-0").removeClass('text-success');
                    $("#tips-0").addClass('text-danger');
                    $("#loading-0").hide(100);
                    $("#upload").show(500);
                } else {
                    layer.msg('错误');
                    $("#loading-0").hide(100);
                    $("#upload").show(500);
                }
            },
            error: function (data) {
            	layer.msg('错误');
            	$("#loading-0").hide(100);
                $("#upload").show(500);
            	console.log(data);
            },
        });
    },
    //将列表数据格式化并显示在页面上
    roundListData: function (id, list, length) {
        var htmlStr = '[ '
        for (var i = 0; i < length; i++) {
            htmlStr += list[i]
            htmlStr += ', '
        }
        htmlStr += ' ... ]'
        $(id).html(htmlStr);
    },
    //获取原始数据
    getOriginData: function () {
    	$.ajax({
            type: 'POST',
            url: '/origin_data',
            dataType: 'json',
            data:{param:'old'},
            headers: {
                'X-CSRFToken': csrftoken
            },//请求
            success: function (data) {//成功接收服务器返回的数据data
            	if (data.code == 1) {
            		layer.msg(data.msg);//显示消息框，展示data
            		$("#loading-1").hide();
            		$("#tips-1").show(500);
            		let list = data.data;
            		info.roundListData("#prData", list, 6);//展示前6个元素
            		$("#click-data").show(1000);//如果选择了查看更多，展示所有元素512个
            		info.roundListData("#faceData", list, 512);
            	} else {
            		layer.msg(data.msg);
            		$("#loading-1").hide(100);
            		$("#tips-1").show(500);
            	}
            },
            error: function (data) {//请求发生错误
            	layer.msg('获取失败');
            	$("#loading-1").hide(100);
            	$("#get-data").show(500);
            	console.log(data);
            },
        });
    },
    getOriginData1: function () {
    	$.ajax({
            type: 'POST',
            url: '/origin_data',
            dataType: 'json',
            data:{param:'new'},
            headers: {
                'X-CSRFToken': csrftoken
            },//请求
            success: function (data) {//成功接收服务器返回的数据data
            	if (data.code == 1) {
            		layer.msg(data.msg);//显示消息框，展示data
            		$("#loading-3").hide();
            		$("#tips-3").show(500);
            		let list = data.data;
            		info.roundListData("#prData1", list, 6);//展示前6个元素
            		$("#click-data1").show(1000);//如果选择了查看更多，展示所有元素512个
            		info.roundListData("#faceData1", list, 512);
            	} else {
            		layer.msg(data.msg);
            		$("#loading-3").hide(100);
            		$("#tips-3").show(500);
            	}
            },
            error: function (data) {//请求发生错误
            	layer.msg('获取失败');
            	$("#loading-3").hide(100);
            	$("#get-data1").show(500);
            	console.log(data);
            },
        });
    },
    getEncryptData: function() {
    	$.ajax({
            type: 'POST',
            url: '/encrypt_data',
            dataType: 'json',
            data:{param:'old'},
            headers: {
                'X-CSRFToken': csrftoken
            },
            success: function (data) {
            	if (data.code == 1) {
            		layer.msg(data.msg);
            		$("#loading-2").hide();
            		$("#tips-2").show(500);
            		$("#click-en-data").show(1000);
            		$("#prEnData").html(data.data.substring(0, 125) + ' ...');
            		$("#prEnData").show(1000);
            		$("#enData").html(data.data);
            	} else {
            		layer.msg(data.msg);
            		$("#loading-2").hide(100);
            		$("#get-en-data").show(1000);
            	}
            },
            error: function (data) {
            	layer.msg('错误');
            	$("#loading-2").hide(100);
            	$("#get-en-data").show(500);
            	console.log(data);
            },
        });
    },
    getEncryptData1: function() {
    	$.ajax({
            type: 'POST',
            url: '/encrypt_data',
            dataType: 'json',
            data:{param:'new'},
            headers: {
                'X-CSRFToken': csrftoken
            },
            success: function (data) {
            	if (data.code == 1) {
            		layer.msg(data.msg);
            		$("#loading-4").hide();
            		$("#tips-4").show(500);
            		$("#click-en-data1").show(1000);
            		$("#prEnData1").html(data.data.substring(0, 125) + ' ...');
            		$("#prEnData1").show(1000);
            		$("#enData1").html(data.data);
            	} else {
            		layer.msg(data.msg);
            		$("#loading-4").hide(100);
            		$("#get-en-data1").show(1000);
            	}
            },
            error: function (data) {
            	layer.msg('错误');
            	$("#loading-4").hide(100);
            	$("#get-en-data1").show(500);
            	console.log(data);
            },
        });
    }
}