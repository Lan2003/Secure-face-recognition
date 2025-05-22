/**
 * 登录
 */

'use strict';

const time = document.querySelector('#time');
$(document).ready(function() {
  $("#gain1").fadeTo("slow", 1.0);
  time.onload = setInterval(() => time.innerHTML = new Date().toLocaleString().slice(9, 19));
})


const upCount = document.querySelector('#uploadCount1');
const csrftoken = $('meta[name=csrf-token]').attr('content')
var uploadCount1 = 0,
  uploadList = [];

/**
 * 照片上传定义
 * @type WeUI
 */
var uploads = weui.uploader('#uploader1', {
  url: '/file_upload1',
  auto: false,
  type: 'file',
  fileVal: 'file',
  compress: {
    width: 1600,
    height: 1600,
    quality: .8
  },
  onBeforeQueued: function(files) {
    // `this` 是轮询到的文件, `files` 是所有文件
    if (["image/jpg", "image/jpeg", "image/png"].indexOf(this.type) < 0) {
      weui.alert('请上传jpg或png格式的图片');
      return false; // 阻止文件添加
    }
    if (this.size > 6 * 1024 * 1024) {
      weui.alert('请上传不超过6M的图片');
      return false;
    }
    if (files.length > 1) { // 防止一下子选择过多文件
      weui.alert('最多只能上传1张图片，请重新选择');
      return false;
    }
    if (uploadCount1 + 1 > 1) {
      weui.alert('最多只能上传1张图片');
      return false;
    }
    ++uploadCount1;
    // return true; // 阻止默认行为，不插入预览图的框架
  },
  onQueued: function() {
    // console.log(this);
    uploadList.push(this);
    upCount.innerHTML = uploadCount1;
    // this.upload();
  },
  onBeforeSend: function(data, headers) {
    $.extend(headers, { 'X-CSRFToken': csrftoken });
    $.extend(headers, { 'Type': 1 });
    // console.log(this, data, headers);
    // $.extend(data, { test: 1 }); // 可以扩展此对象来控制上传参数
    // $.extend(headers, { Origin: 'http://127.0.0.1' }); // 可以扩展此对象来控制上传头部
    // return false; // 阻止文件上传
  },
  onProgress: function(procent) {
    // console.log(this, procent);
    // return true; // 阻止默认行为，不使用默认的进度显示
  },
  onSuccess: function(ret) {
    if (ret.code == 1) {
      layer.msg(ret.msg);
      Cookie.set('user_id', ret.user_id, 24 * 3); // 3天有效 改 原7天
      setTimeout(function() {
        window.location.href = '/info';
      }, 1000);
    } else if (ret.code == 0) {
      layer.msg(ret.msg);
    } else {
      layer.msg('未知错误')
    }
    // console.log(this, ret);
    // return true; // 阻止默认行为，不使用默认的成功态
  },
  onError: function(err) {
    layer.msg('失败');
    // console.log(this, err);
    // return true; // 阻止默认行为，不使用默认的失败态
  }
});

/**
 * 缩略图预览
 */
document.querySelector('#uploaderFiles1').addEventListener('click', function(e) {
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
      weui.confirm('确定删除该图片？', function() {
        --uploadCount1;
        upCount.innerHTML = uploadCount1;
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
const uploadBtn = document.getElementById('upload1');
uploadBtn.addEventListener('click', function() {
  weui.confirm('确认上传？', function() {
      if (uploadList.length === 1 && uploadCount1 === 1) {
        uploadList.forEach(function(file) {
          file.upload();
        });
      } else {
        layer.msg('请选择一张照片');
      }
    },
    function() {
      console.log('no')
    });
});

/**
 * 摄像定义
 * @type WebRTC
 */
const start = document.getElementById('startCamera1'); // 开启摄像头
const back = document.getElementById('back1'); // 返回按钮
const video = document.getElementById('camera1'); // 视频显示区域
const about = document.getElementById('about1');
const canvas = window.canvas = document.querySelector('#canvasi canvas');
const constraints = {
  audio: false,
  video: true
};
canvas.width = 480;
var photoWidth = 480;
var gain = {
  init: function() {
    if (document.body.clientWidth <= 420) {
      photoWidth = document.body.clientWidth - 20 + 'px';
    } else {
      photoWidth = document.getElementById('camera1').offsetWidth;
    }
  }
}

gain.init();

function handleSuccess(stream) {
  $('#start1').hide(500); // 摄像模块区域
  $('#start-i1').hide(500); // 摄像图标
  $('#startCamera1').hide(500); // 开启按钮
  $('#uploadArea1').hide(500); // 照片上传区域
  $('#capture1').show(500); // 拍照按钮
  $('#camera1').show(500); // 视频显示区
  window.stream = stream;
  video.srcObject = stream;
}

function handleError(error) {
  layer.msg('未找到摄像头');
  console.log('navigator.MediaDevices.getUserMedia error: ', error.message, error.name);
}

start1.onclick = function() {
  try {
    navigator.mediaDevices.getUserMedia(constraints).then(handleSuccess).catch(handleError);
  } catch (err) {
    console.log(err.message);
    layer.msg('摄像头开启失败');
  }
};

about1.onclick = function() {
  layer.alert('摄像头捕获依赖于WebRTC，请确保您的设备有摄像头并且支持WebRTC', {
    skin: 'layui-layer-molv',
    title: '关于',
    closeBtn: 0,
  });
};

/**
 * 返回按钮监听
 */
back1.onclick = function() {
  stream.getTracks()[0].stop(); // 关闭摄像头
  $('#start1').show(500);
  $('#start-i1').fadeIn(3000);
  $('#startCamera1').show(500);
  $('#uploadArea1').show(500);
  $('#capture1').hide(500);
  $('#camera1').hide(500);

};

var container = document.getElementById('userimg');

/**
 * 拍照按钮监听
 */
snap1.onclick = function() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  setTimeout(function() {
    document.querySelector('.layui-layer-btn1').style.color = '#000';
  }, 100);
  var windows = layer.open({
    type: 1,
    title: false,
    closeBtn: 0,
    area: photoWidth,
    btn: ['确认', '取消'],
    btnAlign: 'c',
    shadeClose: true,
    content: $('#photo1'),
    yes: function() {
      layer.load(1);
      var img = canvas.toDataURL();
      $.ajax({
        type: 'POST',
        url: '/img_upload1',
        dataType: 'json',
        headers: {
          'X-CSRFToken': csrftoken,
          'Type': 1,
        },
        data: {
          image: img
        },
        success: function(data) {

          layer.closeAll('loading');
          if (data.code == 1) {
            layer.msg(data.msg);
            
            Cookie.set('user_id', data.user_id, 24 * 3); // 3天有效
            setTimeout(function() {
              window.location.href = '/info';//跳转到/info（不是info.html）
            }, 1000);
          } else if (data.code == 0) {
            layer.msg(data.msg, { time: 4000 });
          } else {
            layer.msg('未知错误')
          }
          setTimeout(function() {
            layer.closeAll();
          }, 2000);
        },
        error: function(date) {
          layer.msg('失败');
          layer.closeAll();
        }
      });
    }
  });
};

/**
 * 初始加载按钮监听

$("#load").on('click', function () {
  $.ajax({
      type: 'POST',
      url: '/load',
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
});*/