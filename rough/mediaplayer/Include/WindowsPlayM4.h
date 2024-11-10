#ifndef _WINDOWS_PLAYM4_H_
#define _WINDOWS_PLAYM4_H_

#if defined( _WINDLL)
    #define PLAYM4_API  extern "C" __declspec(dllexport)
#else 
    #define PLAYM4_API  extern "C" __declspec(dllimport)
#endif

//Max channel numbers
#define PLAYM4_MAX_SUPPORTS 500
//Wave coef range;
#define MIN_WAVE_COEF -100
#define MAX_WAVE_COEF 100

//Timer type
#define TIMER_1 1 //Only 16 timers for every process.Default TIMER;
#define TIMER_2 2 //Not limit;But the precision less than TIMER_1;

//BUFFER AND DATA TYPE
#define BUF_VIDEO_SRC               (1) //mixed input,total src buffer size;splited input,video src buffer size 
#define BUF_AUDIO_SRC               (2) //mixed input,not defined;splited input,audio src buffer size
#define BUF_VIDEO_RENDER            (3) //video render node count 
#define BUF_AUDIO_RENDER            (4) //audio render node count 
#define BUF_VIDEO_DECODED           (5) //video decoded node count to render
#define BUF_AUDIO_DECODED           (6) //audio decoded node count to render
#define BUF_VIDEO_SRC_EX            (7) //video render node count without hangnode;resetbuf need

//Error code
#define  PLAYM4_NOERROR                         0   //no error
#define  PLAYM4_PARA_OVER                       1   //input parameter is invalid;
#define  PLAYM4_ORDER_ERROR                     2   //The order of the function to be called is error.
#define  PLAYM4_TIMER_ERROR                     3   //Create multimedia clock failed;
#define  PLAYM4_DEC_VIDEO_ERROR                 4   //Decode video data failed.
#define  PLAYM4_DEC_AUDIO_ERROR                 5   //Decode audio data failed.
#define  PLAYM4_ALLOC_MEMORY_ERROR              6   //Allocate memory failed.
#define  PLAYM4_OPEN_FILE_ERROR                 7   //Open the file failed.
#define  PLAYM4_CREATE_OBJ_ERROR                8   //Create thread or event failed
#define  PLAYM4_CREATE_DDRAW_ERROR              9   //Create DirectDraw object failed.
#define  PLAYM4_CREATE_OFFSCREEN_ERROR          10  //failed when creating off-screen surface.
#define  PLAYM4_BUF_OVER                        11  //buffer is overflow
#define  PLAYM4_CREATE_SOUND_ERROR              12  //failed when creating audio device.	
#define  PLAYM4_SET_VOLUME_ERROR                13  //Set volume failed
#define  PLAYM4_SUPPORT_FILE_ONLY               14  //The function only support play file.
#define  PLAYM4_SUPPORT_STREAM_ONLY             15  //The function only support play stream.
#define  PLAYM4_SYS_NOT_SUPPORT                 16  //System not support.
#define  PLAYM4_FILEHEADER_UNKNOWN              17  //No file header.
#define  PLAYM4_VERSION_INCORRECT               18  //The version of decoder and encoder is not adapted.  
#define  PLAYM4_INIT_DECODER_ERROR              19  //Initialize decoder failed.
#define  PLAYM4_CHECK_FILE_ERROR                20  //The file data is unknown.
#define  PLAYM4_INIT_TIMER_ERROR                21  //Initialize multimedia clock failed.
#define  PLAYM4_BLT_ERROR                       22  //Blt failed.
#define  PLAYM4_UPDATE_ERROR                    23  //Update failed.
#define  PLAYM4_OPEN_FILE_ERROR_MULTI           24  //openfile error, streamtype is multi
#define  PLAYM4_OPEN_FILE_ERROR_VIDEO           25  //openfile error, streamtype is video
#define  PLAYM4_JPEG_COMPRESS_ERROR             26  //JPEG compress error
#define  PLAYM4_EXTRACT_NOT_SUPPORT             27  //Don't support the version of this file.
#define  PLAYM4_EXTRACT_DATA_ERROR              28  //extract video data failed.
#define  PLAYM4_SECRET_KEY_ERROR                29  //Secret key is error //add 20071218
#define  PLAYM4_DECODE_KEYFRAME_ERROR           30  //add by hy 20090318
#define  PLAYM4_NEED_MORE_DATA                  31  //add by hy 20100617
#define  PLAYM4_INVALID_PORT                    32  //add by cj 20100913
#define  PLAYM4_NOT_FIND                        33  //add by cj 20110428
#define  PLAYM4_NEED_LARGER_BUFFER              34  //add by pzj 20130528
#define  PLAYM4_FAIL_UNKNOWN                    99  //Fail, but the reason is unknown;	

//鱼眼功能错误码
#define PLAYM4_FEC_ERR_ENABLEFAIL               100 // 鱼眼模块加载失败
#define PLAYM4_FEC_ERR_NOTENABLE                101 // 鱼眼模块没有加载
#define PLAYM4_FEC_ERR_NOSUBPORT                102 // 子端口没有分配
#define PLAYM4_FEC_ERR_PARAMNOTINIT             103 // 没有初始化对应端口的参数
#define PLAYM4_FEC_ERR_SUBPORTOVER              104 // 子端口已经用完
#define PLAYM4_FEC_ERR_EFFECTNOTSUPPORT         105 // 该安装方式下这种效果不支持
#define PLAYM4_FEC_ERR_INVALIDWND               106 // 非法的窗口
#define PLAYM4_FEC_ERR_PTZOVERFLOW              107 // PTZ位置越界
#define PLAYM4_FEC_ERR_RADIUSINVALID            108 // 圆心参数非法
#define PLAYM4_FEC_ERR_UPDATENOTSUPPORT         109 // 指定的安装方式和矫正效果，该参数更新不支持
#define PLAYM4_FEC_ERR_NOPLAYPORT               110 // 播放库端口没有启用
#define PLAYM4_FEC_ERR_PARAMVALID               111 // 参数为空
#define PLAYM4_FEC_ERR_INVALIDPORT              112 // 非法子端口
#define PLAYM4_FEC_ERR_PTZZOOMOVER              113 // PTZ矫正范围越界
#define PLAYM4_FEC_ERR_OVERMAXPORT              114 // 矫正通道饱和，最大支持的矫正通道为64个
#define PLAYM4_FEC_ERR_ENABLED                  115 // 该端口已经启用了鱼眼模块
#define PLAYM4_FEC_ERR_D3DACCENOTENABLE         116 // D3D加速没有开启
#define PLAYM4_FEC_ERR_PLACETYPE                117 // 安装方式不对.一个播放库port，对应一种安装方式
#define PLAYM4_FEC_ERR_CorrectType              118 // 矫正方式错误：如矫正方式已有,则不能开多个；比如一个播放库port,设定除了PTZ和鱼眼半球矫正方式外,其他的矫正方式则只能开一路;设置180度矫正不能和ptz矫正一起开，半球矫正与其他矫正则无关联性。
#define PLAYM4_FEC_ERR_NULLWND                  119 // 鱼眼窗口为空
#define PLAYM4_FEC_ERR_PARA                     120 // 鱼眼参数错误


//Max display regions.
#define MAX_DISPLAY_WND     4

//Display type
#define DISPLAY_NORMAL      0x00000001
#define DISPLAY_QUARTER     0x00000002
#define DISPLAY_YC_SCALE    0x00000004
#define DISPLAY_NOTEARING   0x00000008

//Display buffers
#define MAX_DIS_FRAMES      50
#define MIN_DIS_FRAMES      1

//Locate by
#define BY_FRAMENUM         1
#define BY_FRAMETIME        2

//Source buffer
#define SOURCE_BUF_MAX      1024*100000
#define SOURCE_BUF_MIN      1024*50

//Stream type
#define STREAME_REALTIME    0
#define STREAME_FILE        1

//frame type
#define T_AUDIO16           101
#define T_AUDIO8            100
#define T_UYVY              1
#define T_YV12              3
#define T_RGB32             7

//capability
#define SUPPORT_DDRAW       1
#define SUPPORT_BLT         2
#define SUPPORT_BLTFOURCC   4
#define SUPPORT_BLTSHRINKX  8
#define SUPPORT_BLTSHRINKY  16
#define SUPPORT_BLTSTRETCHX 32
#define SUPPORT_BLTSTRETCHY 64
#define SUPPORT_SSE         128
#define SUPPORT_MMX         256

// 以下宏定义用于HIK_MEDIAINFO结构
#define FOURCC_HKMI         0x484B4D49  // "HKMI" HIK_MEDIAINFO结构标记

// 系统封装格式
#define SYSTEM_NULL         0x0         // 没有系统层，纯音频流或视频流
#define SYSTEM_HIK          0x1         // 海康文件层
#define SYSTEM_MPEG2_PS     0x2         // PS封装
#define SYSTEM_MPEG2_TS     0x3         // TS封装
#define SYSTEM_RTP          0x4         // rtp封装
#define SYSTEM_RTMP         0xD         // rtmp封装
#define SYSTEM_RTPHIK       0x401       // rtp封装hik
#define SYSTEM_RTP_JT       0x104       // rtpjt封装
#define SYSTEM_DAH          0x8001      // 大华封装

// 视频编码类型
#define VIDEO_NULL          0x0 // 没有视频
#define VIDEO_H264          0x1 // 海康H.264
#define VIDEO_MPEG2         0x2 // 标准MPEG2
#define VIDEO_MPEG4         0x3 // 标准MPEG4
#define VIDEO_MJPEG         0x4
#define VIDEO_AVC265        0x5 // 标准H265/AVC
#define VIDEO_SVAC          0x6
#define VIDEO_AVC264        0x0100

// 音频编码类型
#define AUDIO_NULL          0x0000 // 没有音频
#define AUDIO_ADPCM         0x1000 // ADPCM
#define AUDIO_MPEG          0x2000 // MPEG 系列音频，解码器能自适应各种MPEG音频
#define AUDIO_AAC           0X2001 // AAC 编码
#define AUDIO_RAW_DATA8     0x7000 //采样率为8k的原始数据
#define AUDIO_RAW_UDATA16   0x7001 //采样率为16k的原始数据，即L16
// G系列音频
#define AUDIO_RAW_DATA8     0x7000      //采样率为8k的原始数据
#define AUDIO_RAW_UDATA16   0x7001      //采样率为16k的原始数据，即L16
#define AUDIO_G711_U        0x7110
#define AUDIO_G711_A        0x7111
#define AUDIO_G722_1        0x7221
#define AUDIO_G723_1        0x7231
#define AUDIO_G726_U        0x7260
#define AUDIO_G726_A        0x7261
#define AUDIO_G726_16       0x7262
#define AUDIO_G729          0x7290
#define AUDIO_AMR_NB        0x3000

#define SYNCDATA_VEH        1 //同步数据:车载信息
#define SYNCDATA_IVS        2 //同步数据:智能信息

//motion flow type
#define	MOTION_FLOW_NONE    0
#define MOTION_FLOW_CPU     1
#define MOTION_FLOW_GPU     2

//音视频加密类型
#define ENCRYPT_AES_3R_VIDEO     1
#define ENCRYPT_AES_10R_VIDEO    2
#define ENCRYPT_AES_3R_AUDIO     1
#define ENCRYPT_AES_10R_AUDIO    2

///<旋转角度
#define R_ANGLE_0           -1      //不旋转
#define R_ANGLE_L90         0       //向左旋转90度
#define R_ANGLE_R90         1       //向右旋转90度
#define R_ANGLE_180         2       //旋转180度


//Frame position
typedef struct{
    LONGLONG nFilePos;
    long nFrameNum;
    long nFrameTime;
    long nErrorFrameNum;
    SYSTEMTIME *pErrorTime;
    long nErrorLostFrameNum;
    long nErrorFrameSize;
}FRAME_POS,*PFRAME_POS;

//Frame Info
typedef struct{
    long nWidth;
    long nHeight;
    long nStamp;
    long nType;
    long nFrameRate;
    DWORD dwFrameNum;
}FRAME_INFO;

//Frame 
typedef struct{
    char *pDataBuf;
    long  nSize;
    long  nFrameNum;
    BOOL  bIsAudio;
    long  nReserved;
}FRAME_TYPE;

//Watermark Info
typedef struct{
    char *pDataBuf;
    long  nSize;
    long  nFrameNum;
    BOOL  bRsaRight;
    long  nReserved;
}WATERMARK_INFO;

typedef struct SYNCDATA_INFO 
{
    DWORD dwDataType;       //和码流数据同步的附属信息类型，目前有：智能信息，车载信息
    DWORD dwDataLen;        //附属信息数据长度
    BYTE* pData;            //指向附属信息数据结构的指针,比如IVS_INFO结构
} SYNCDATA_INFO;

typedef struct _VCA_RECT_F_
{
    float x;         //矩形左上角X轴坐标
    float y;         //矩形左上角Y轴坐标
    float width;     //矩形宽度
    float height;    //矩形高度
}VCA_RECT_F;

// 扩展信息结构体
typedef struct
{
    unsigned char        privt_type;                     // 用户扩展类型:详见IVS_PRIVT_INFO，10表示颜色+违禁品类型
    unsigned char        reseverd[6];                    // 目标类型扩展
    unsigned char        privt_len;                      // 用户扩展信息长度
    unsigned char        privt_data[32];                 // 用户扩展信息
}IS_PRIVT_INFO;

// 颜色结构体
typedef struct
{
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    unsigned char alpha;
}IS_PRIVT_INFO_COLOR;

// 违禁物颜色结构体
typedef struct
{
    IS_PRIVT_INFO_COLOR color;                  // 颜色
    unsigned char       confidence;             // 违禁品置信度
    unsigned char       pos_len;                // 违禁品名称长度
    unsigned char       pos_data[22];           // 违禁品名称
    unsigned int        type;                   // 违禁品类型
}IS_PRIVT_INFO_CONTRABAND;

//目标信息结构体
typedef struct _VCA_TARGET_EX
{
    unsigned int      ID;          //ID
    unsigned char     reserved[8]; //保留字节
    VCA_RECT_F        rect;        //目标框
    unsigned char     reserved1[40]; //私有信息扩展字段，暂时以unsigned char数组表示，如果外部有需求再告知具体结构与信息
}VCA_TARGET_EX;

typedef struct _VCA_TARGET_LIST_EX
{
    unsigned int    target_num;             //目标数量
    VCA_TARGET_EX   *pstTarget;             //目标内容，按照VCA_TARGET_EX的大小进行偏移读取各个目标具体的信息即可
}VCA_TARGET_LIST_EX;

//智能信息结构体回调用
typedef struct _INTEL_INFO_EX
{
    unsigned int                   type;               ///< 标记回调出了哪些私有信息类型
    VCA_TARGET_LIST_EX             stTarget;           ///< 目标
    VCA_TARGET_LIST_EX             stTarget_EX;        ///< 深眸目标
}INTEL_INFO_EX;

/////////////////////////////////////////////////////////////////////////////////////////////
//IVS新智能信息回调接口结构体定义
/////////////////////////////////////////////////////////////////////////////////////////////

///                                目标框                           ///
typedef struct _VCA_POINT_F_
{
    float x;
    float y;
}VCA_POINT_F;

//多边形(浮点型)
typedef struct _VCA_POLYGON_F_
{
    unsigned int  vertex_num;                  //顶点数
    VCA_POINT_F   point[10];   //顶点
}VCA_POLYGON_F;

//旋转矩形
typedef struct _VCA_ROTATE_RECT_F_
{
    float				  cx;						// 矩形中心点X轴坐标
    float				  cy;						// 矩形中心点Y轴坐标
    float				  width;					// 矩形宽度
    float				  height;					// 矩形高度
    float				  theta;				   // 旋转矩形角度
}VCA_ROTATE_RECT_F;

//区域联合体
typedef struct _VCA_REGION_
{
    unsigned int region_type;           // 参考VCA_REGION_TYPE，2表示多边形，3表示矩形
    char         reserved[12];
    union
    {
        unsigned char		size[84];
        VCA_POLYGON_F       polygon;                // 多边形
        VCA_RECT_F          rect;                   // 矩形
        VCA_ROTATE_RECT_F 	rotate_rect;	 		 // 旋转矩形,暂不支持
    };
}VCA_REGION;

//单个目标框结构体
typedef struct 
{
    unsigned int            id;
    unsigned int            blob_type;        // 目标类_OBJ_TYPE
    short                   confidence;       // 目标框置信度
    char                    reserved[14];
    VCA_REGION              region;           // 目标位置区域
    unsigned char           privt_info[40];   // 扩展信息
}HIK_TARGET_BLOB_EX;

//目标框列表信息定义：
typedef struct _VCA_TARGET_LIST_V1_EX_
{
    unsigned int         LineType;      //框类型，0表示矩形框，1表示四角框
    unsigned int         target_num;    //目标数量
    HIK_TARGET_BLOB_EX   *pstTarget;    //目标内容，按照HIK_TARGET_BLOB_EX的大小进行偏移读取各个目标具体的信息即可
}VCA_TARGET_LIST_V1_EX;

///                                规则框                           ///
//单个规则框结构体
typedef struct _VCA_RULE_EX
{
    unsigned char       ID;               //规则ID
    unsigned char       reserved[15];       //保留字节
    VCA_POLYGON_F       polygon;         //规则对应的多边形区域
    unsigned char       privt_info[40];      // 扩展信息
}VCA_RULE_EX;

//规则框列表信息定义：
typedef struct _VCA_RULE_LIST_V3_EX_
{
    unsigned int  LineType;      //框类型，0表示矩形框，1 表示四角框
    unsigned int  rule_num;     //链表中规则数量
    VCA_RULE_EX   *pstRule;   //规则内容，按照VCA_RULE_EX的大小进行偏移读取各个目标具体的信息即可
}VCA_RULE_LIST_V3_EX;

///                                报警框                           ///

typedef struct  _VCA_ALERT_EX_
{
    unsigned char     alert;           //有无报警信息: 0-没有，1-有
    unsigned char     reserved[7];     //保留字节
    VCA_RULE_EX       rule_info;       //报警规则信息
    VCA_TARGET_EX        target;          //报警目标信息
    unsigned char     privt_info[40];  // 扩展信息
}VCA_ALERT_EX;

//报警信息定义：
typedef struct  _VCA_ALERT_LIST_EX_
{
    unsigned int           alert_num;  //报警数量
    VCA_ALERT_EX           *pstAlert; 
}VCA_ALERT_LIST_EX;

//智能信息结构体回调用
typedef struct _PRIVATE_INFO_
{
    unsigned int                  type;    ///< 标记回调出了哪些私有信息类型
    VCA_TARGET_LIST_V1_EX         stTarget;           ///< 普通目标框，类型 1
    VCA_TARGET_LIST_V1_EX         stTarget_EX;        ///< 深眸目标框，类型 2
    VCA_RULE_LIST_V3_EX           stRule;             ///< 普通规则框，类型 4
    VCA_RULE_LIST_V3_EX           stRule_EX;          ///< 深眸规则框，类型 8
    VCA_ALERT_LIST_EX             stAlert;            ///< 报警信息，类型 16
} PRIVATE_INFO;

/////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _HIK_MEDIAINFO_FLAG_
#define _HIK_MEDIAINFO_FLAG_
typedef struct _HIK_MEDIAINFO_
{
    unsigned int    media_fourcc;           // "HKMI": 0x484B4D49 Hikvision Media Information
    unsigned short  media_version;          // 版本号：指本信息结构版本号，目前为0x0101,即1.01版本，01：主版本号；01：子版本号。
    unsigned short  device_id;              // 设备ID，便于跟踪/分析
    unsigned short  system_format;          // 系统封装层
    unsigned short  video_format;           // 视频编码类型
    unsigned short  audio_format;           // 音频编码类型
    unsigned char   audio_channels;         // 通道数  
    unsigned char   audio_bits_per_sample;  // 样位率
    unsigned int    audio_samplesrate;      // 采样率 
    unsigned int    audio_bitrate;          // 压缩音频码率,单位：bit
    unsigned int    reserved[4];            // 保留
}HIK_MEDIAINFO;
#endif

typedef struct
{
    long nPort;
    char * pBuf;
    long nBufLen;
    long nWidth;
    long nHeight;
    long nStamp;
    long nType;
    void* nUser;
}DISPLAY_INFO;

typedef struct
{
    long nPort;
    char *pVideoBuf;
    long nVideoBufLen;
    char *pPriBuf;
    long nPriBufLen;
    long nWidth;
    long nHeight;
    long nStamp;
    long nType;
    void* nUser;
}DISPLAY_INFOEX;

typedef struct
{
    long         nPort;        //通道号
    char         *pBuf;        //返回的第一路图像数据指针
    unsigned int nBufLen;      //返回的第一路图像数据大小
    char         *pBuf1;       //返回的第二路图像数据指针
    unsigned int nBufLen1;     //返回的第二路图像数据大小
    char         *pBuf2;       //返回的第三路图像数据指针
    unsigned int nBufLen2;     //返回的第三路图像数据大小
    unsigned int nWidth;       //画面宽
    unsigned int nHeight;      //画面高
    unsigned int nStamp;       //时标信息，单位毫秒
    unsigned int nType;        //数据类型
    void         *pUser;       //用户数据
    unsigned int reserved[4];  //保留
}DISPLAY_INFO_YUV;

typedef struct PLAYM4_SYSTEM_TIME //绝对时间
{
    DWORD dwYear;   //年
    DWORD dwMon;    //月
    DWORD dwDay;    //日
    DWORD dwHour;   //时
    DWORD dwMin;    //分
    DWORD dwSec;    //秒
    DWORD dwMs;     //毫秒
} PLAYM4_SYSTEM_TIME;

#ifndef CROP_PIC_INFO_TAG
#define CROP_PIC_INFO_TAG
typedef struct
{
    BYTE  *pDataBuf;      //抓图数据buffer
    DWORD dwPicSize;      //实际图片大小
    DWORD dwBufSize;      //数据buffer大小
    DWORD dwPicWidth;     //截图宽
    DWORD dwPicHeight;    //截图高
    DWORD dwReserve;      //多加一个reserve字段
    RECT  *pCropRect;     //选择区域NULL, 同老的抓图接口
}CROP_PIC_INFO;
#endif

//ENCRYPT Info
typedef struct{
    long nVideoEncryptType;  //视频加密类型
    long nAudioEncryptType;  //音频加密类型
    long nSetSecretKey;      //是否设置，1表示设置密钥，0表示没有设置密钥
}ENCRYPT_INFO;

///<私有信息模块类型
typedef enum _PLAYM4_PRIDATA_RENDER
{
    PLAYM4_RENDER_ANA_INTEL_DATA    = 0x00000001, //智能分析
    PLAYM4_RENDER_MD                = 0x00000002, //移动侦测
    PLAYM4_RENDER_ADD_POS           = 0x00000004, //POS信息后叠加
    PLAYM4_RENDER_ADD_PIC           = 0x00000008, //图片叠加信息
    PLAYM4_RENDER_FIRE_DETCET       = 0x00000010, //热成像信息
    PLAYM4_RENDER_TEM               = 0x00000020, //温度信息
    PLAYM4_RENDER_TRACK_TEM         = 0x00000040, //轨迹信息
    PLAYM4_RENDER_THERMAL           = 0x00000080, //废气检测和烟火屏蔽信息
}PLAYM4_PRIDATA_RENDER;

typedef enum _PLAYM4_THERMAL_FLAG
{
    PLAYM4_THERMAL_FIREMASK           = 0x00000001, //烟火屏蔽
    PLAYM4_THERMAL_RULEGAS            = 0x00000002, //规则废气检测
    PLAYM4_THERMAL_TARGETGAS          = 0x00000004, //目标废气检测
}PLAYM4_THERMAL_FLAG;

typedef enum _PLAYM4_FIRE_ALARM
{
    PLAYM4_FIRE_FRAME_DIS           = 0x00000001, //火点框显示
    PLAYM4_FIRE_MAX_TEMP            = 0x00000002, //最高温度
    PLAYM4_FIRE_MAX_TEMP_POSITION   = 0x00000004, //最高温度位置显示
    PLAYM4_FIRE_DISTANCE            = 0x00000008, //最高温度距离
}PLAYM4_FIRE_ALARM;

typedef enum _PLAYM4_TEM_FLAG
{
    PLAYM4_TEM_REGION_BOX             = 0x00000001, //框测温
    PLAYM4_TEM_REGION_LINE            = 0x00000002, //线测温
    PLAYM4_TEM_REGION_POINT           = 0x00000004, //点测温
}PLAYM4_TEM_FLAG;

typedef enum _PLAYM4_TRACK_FLAG
{
    PLAYM4_TRACK_PEOPLE               = 0x00000001, //人轨迹
    PLAYM4_TRACK_VEHICLE              = 0x00000002, //车轨迹
}PLAYM4_TRACK_FLAG;

/////////////////////////////////////////////////////////////////////////////////////////////
//Proposed Interface 建议使用的接口
/////////////////////////////////////////////////////////////////////////////////////////////
///<初始化
PLAYM4_API BOOL __stdcall PlayM4_GetPort(LONG* nPort);
PLAYM4_API BOOL __stdcall PlayM4_FreePort(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_OpenFile(LONG nPort,LPSTR sFileName);
PLAYM4_API BOOL __stdcall PlayM4_CloseFile(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_OpenStream(LONG nPort,PBYTE pFileHeadBuf,DWORD nSize,DWORD nBufPoolSize);
PLAYM4_API BOOL __stdcall PlayM4_CloseStream(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_InputData(LONG nPort,PBYTE pBuf,DWORD nSize);
PLAYM4_API BOOL __stdcall PlayM4_SetStreamOpenMode(LONG nPort,DWORD nMode);
PLAYM4_API LONG __stdcall PlayM4_GetStreamOpenMode(LONG nPort);

///<索引
PLAYM4_API BOOL __stdcall PlayM4_SetFileRefCallBack(LONG nPort, void (__stdcall *pFileRefDone)(DWORD nPort,void* nUser),void* nUser);
PLAYM4_API BOOL __stdcall PlayM4_GetRefValue(LONG nPort,BYTE *pBuffer, DWORD *pSize);
PLAYM4_API BOOL __stdcall PlayM4_SetRefValue(LONG nPort,BYTE *pBuffer, DWORD nSize);
PLAYM4_API BOOL __stdcall PlayM4_GetRefValueEx(LONG nPort,BYTE *pBuffer, DWORD *pSize);///< 与PlayM4_GetRefValueEx相比，增加索引帧类型和长度回调
PLAYM4_API BOOL __stdcall PlayM4_GetKeyFramePos(LONG nPort,DWORD nValue, DWORD nType, PFRAME_POS pFramePos);
PLAYM4_API BOOL __stdcall PlayM4_GetNextKeyFramePos(LONG nPort,DWORD nValue, DWORD nType, PFRAME_POS pFramePos);

///<播放控制
PLAYM4_API BOOL __stdcall PlayM4_Play(LONG nPort, HWND hWnd);
PLAYM4_API BOOL __stdcall PlayM4_Stop(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_Pause(LONG nPort,DWORD nPause);
PLAYM4_API BOOL __stdcall PlayM4_Fast(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_Slow(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_OneByOne(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_OneByOneBack(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_ReversePlay(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_RefreshPlay(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_RefreshPlayEx(LONG nPort,DWORD nRegionNum);

///<声音播放
PLAYM4_API BOOL __stdcall PlayM4_PlaySound(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_StopSound();
PLAYM4_API BOOL __stdcall PlayM4_PlaySoundShare(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_StopSoundShare(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_SetVolume(LONG nPort,WORD nVolume);
PLAYM4_API WORD __stdcall PlayM4_GetVolume(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_AdjustWaveAudio(LONG nPort,LONG nCoefficient);
PLAYM4_API BOOL	__stdcall PlayM4_SetANRParam(LONG nPort, BOOL nEnable, int nANRLevel);

///<获取信息
PLAYM4_API BOOL __stdcall PlayM4_SetPlayPos(LONG nPort,float fRelativePos);
PLAYM4_API float __stdcall PlayM4_GetPlayPos(LONG nPort);
PLAYM4_API DWORD __stdcall PlayM4_GetFileTime(LONG nPort);
PLAYM4_API DWORD __stdcall PlayM4_GetPlayedTime(LONG nPort);
PLAYM4_API DWORD __stdcall PlayM4_GetPlayedFrames(LONG nPort);
PLAYM4_API DWORD __stdcall PlayM4_GetFileTotalFrames(LONG nPort);
PLAYM4_API DWORD __stdcall PlayM4_GetCurrentFrameRate(LONG nPort);
PLAYM4_API DWORD __stdcall PlayM4_GetPlayedTimeEx(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_SetPlayedTimeEx(LONG nPort,DWORD nTime);
PLAYM4_API DWORD __stdcall PlayM4_GetCurrentFrameNum(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_SetCurrentFrameNum(LONG nPort,DWORD nFrameNum);
PLAYM4_API BOOL __stdcall PlayM4_GetSystemTime(LONG nPort, PLAYM4_SYSTEM_TIME *pstSystemTime);
PLAYM4_API DWORD __stdcall PlayM4_GetSpecialData(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_GetPictureSize(LONG nPort,LONG *pWidth,LONG *pHeight);
PLAYM4_API BOOL __stdcall PlayM4_SetFileEndCallback(LONG nPort, void(CALLBACK*FileEndCallback)(long nPort, void *pUser), void *pUser);
PLAYM4_API BOOL __stdcall PlayM4_GetFileTotalTime(LONG nPort, PLAYM4_SYSTEM_TIME *pstBegin, PLAYM4_SYSTEM_TIME *pstStop);
///<时区的表示以秒为单位，东区为正，西区为负
PLAYM4_API BOOL __stdcall PlayM4_SetSupplementaryTimeZone(int nPort, int nTimeZone); 
///<若已经设置时区则返回设置时区，否则失败
PLAYM4_API BOOL __stdcall PlayM4_GetSupplementaryTimeZone(int nPort, int* pTimeZone); 
///<获取时区信息
PLAYM4_API int __stdcall PlayM4_GetTimeZoneInfo(long nPort, int* pTimeZone);
//获取封装和编码类型
PLAYM4_API int __stdcall PlayM4_GetStreamInfo(long nPort, int* pSysFormat, int* pCodeType);
///<缓存操作
PLAYM4_API DWORD __stdcall PlayM4_GetSourceBufferRemain(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_ResetSourceBuffer(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_SetSourceBufCallBack(LONG nPort,DWORD nThreShold,void (CALLBACK * SourceBufCallBack)(long nPort,DWORD nBufSize,void* dwUser,void*pResvered),void* dwUser,void *pReserved);
PLAYM4_API BOOL __stdcall PlayM4_ResetSourceBufFlag(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_ResetBuffer(LONG nPort,DWORD nBufType);
PLAYM4_API DWORD __stdcall PlayM4_GetBufferValue(LONG nPort,DWORD nBufType);
PLAYM4_API BOOL __stdcall PlayM4_SetDisplayBuf(LONG nPort,DWORD nNum);
PLAYM4_API DWORD __stdcall PlayM4_GetDisplayBuf(LONG nPort);

///<抓图
PLAYM4_API BOOL __stdcall PlayM4_SetJpegQuality(long nQuality);
PLAYM4_API BOOL __stdcall PlayM4_GetBMP(LONG nPort,PBYTE pBitmap,DWORD nBufSize,DWORD* pBmpSize);
PLAYM4_API BOOL __stdcall PlayM4_GetJPEG(LONG nPort,PBYTE pJpeg,DWORD nBufSize,DWORD* pJpegSize);
PLAYM4_API BOOL __stdcall PlayM4_ConvertToBmpFile(char * pBuf,long nSize,long nWidth,long nHeight,long nType,char *sFileName);
PLAYM4_API BOOL __stdcall PlayM4_ConvertToJpegFile(char * pBuf,long nSize,long nWidth,long nHeight,long nType,char *sFileName);

PLAYM4_API BOOL __stdcall PlayM4_SetIdemuxPara(LONG nPort, int nChunkSize);

///<解码
PLAYM4_API BOOL __stdcall PlayM4_SetSecretKey(LONG nPort, LONG lKeyType, char *pSecretKey, LONG lKeyLen);
PLAYM4_API BOOL __stdcall PlayM4_ThrowBFrameNum(LONG nPort,DWORD nNum);
PLAYM4_API BOOL __stdcall PlayM4_SetDecCBStream(LONG nPort,DWORD nStream);
PLAYM4_API BOOL __stdcall PlayM4_SetDecodeFrameType(LONG nPort,DWORD nFrameType);
PLAYM4_API BOOL __stdcall PlayM4_CheckDiscontinuousFrameNum(LONG nPort, BOOL bCheck);
PLAYM4_API BOOL __stdcall PlayM4_SkipErrorData(LONG nPort, BOOL bSkip);
PLAYM4_API BOOL __stdcall PlayM4_SetDecCallBackMend(LONG nPort,void (CALLBACK* DecCBFun)(long nPort,char * pBuf,long nSize,FRAME_INFO * pFrameInfo, void* nUser,void* nReserved2), void* nUser);
PLAYM4_API BOOL __stdcall PlayM4_SetDecCallBackExMend(LONG nPort, void (CALLBACK* DecCBFun)(long nPort, char* pBuf, long nSize, FRAME_INFO* pFrameInfo, void* nUser, void* nReserved2), char* pDest, long nDestSize, void* nUser);
// 加密码流回调,nType=0表示码流加密标记位发生变化就回调，nType=1表示码流有加密位发生回调-只回调一次
PLAYM4_API BOOL __stdcall PlayM4_SetEncryptTypeCallBack(LONG nPort, DWORD nType, void (CALLBACK* EncryptTypeCBFun)(long nPort, ENCRYPT_INFO* pEncryptInfo, void* nUser, long nReserved2), void* nUser);

//解码或显示回调中绝对时间戳（帧号）或相对时间戳（帧号）的回调设置;
// nModule 表示回调模块选择，0 表示默认解码回调和显示回调 1：表示解码回调，2：表示显示回调；
// nType 表示回调绝对时间戳（帧号）还是相对时间戳（帧号）的类型设置，0：表示回调相对时间戳和相对帧号 1: 表示回调绝对时间戳和绝对帧号
//显示回调是没有帧号回调的，所以显示回调下 0：表示回调相对时间戳 1: 表示回调绝对时间戳
PLAYM4_API BOOL __stdcall PlayM4_SetDecOrDisplayCallbackType(int nPort, int nModule, int nType);

///<显示
PLAYM4_API BOOL __stdcall PlayM4_SetDisplayRegion(LONG nPort,DWORD nRegionNum, RECT *pSrcRect, HWND hDestWnd, BOOL bEnable);///<电子放大
PLAYM4_API BOOL __stdcall PlayM4_SetDisplayRegionOnWnd(LONG nPort,DWORD nRegionNum, RECT *pSrcRect, BOOL bEnable);///<多窗口分割
PLAYM4_API BOOL __stdcall PlayM4_SetDisplayType(LONG nPort,LONG nType);
PLAYM4_API long __stdcall PlayM4_GetDisplayType(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_SetDisplayCallBack(LONG nPort,void (CALLBACK* DisplayCBFun)(long nPort,char * pBuf,long nSize,long nWidth,long nHeight,long nStamp,long nType,void* nReserved));
PLAYM4_API BOOL __stdcall PlayM4_SetDisplayCallBackEx(LONG nPort,void (CALLBACK* DisplayCBFun)(DISPLAY_INFO *pstDisplayInfo), void* nUser);
PLAYM4_API BOOL __stdcall PlayM4_RegisterDrawFun(LONG nPort,void (CALLBACK* DrawFun)(long nPort,HDC hDc,void* nUser),void* nUser);
PLAYM4_API BOOL __stdcall PlayM4_SetEncTypeChangeCallBack(LONG nPort,void(CALLBACK *funEncChange)(long nPort,void* nUser),void* nUser);
PLAYM4_API BOOL __stdcall PlayM4_SetCheckWatermarkCallBack(LONG nPort,void(CALLBACK* funCheckWatermark)(long nPort,WATERMARK_INFO* pWatermarkInfo,void* nUser),void* nUser);
PLAYM4_API BOOL __stdcall PlayM4_RenderPrivateData(LONG nPort, int nIntelType, BOOL bTrue);
PLAYM4_API BOOL __stdcall PlayM4_RenderPrivateDataEx(LONG nPort, int nIntelType, int nSubType, BOOL bTrue);
PLAYM4_API BOOL __stdcall PlayM4_SetOverlayPriInfoFlag(LONG nPort, int nIntelType, BOOL bTrue);
PLAYM4_API BOOL __stdcall PlayM4_SetTrackDurationTime(LONG nPort, int nTime);

//私有信息回调接口，nType表示需要回调的私有信息类型，目前支持两种类型: 普通目标框：1; 深眸目标框； 2; 支持输入 1|2 的形式，即输入3表示两种类型都会回调
PLAYM4_API BOOL __stdcall PlayM4_RegisterIVSDrawFunCB_EX(long nPort, void (CALLBACK* IVSDrawFun)(long nPort, HDC hDC, FRAME_INFO* pFrameInfo, INTEL_INFO_EX* pSyncData, void*  dwUser), void*  dwUser, unsigned int nType);

//lType: 1 表示获取当前显示帧PTZ信息。以特定结构体形式存储在pInfo内，plLen返回长度信息;起先传入pInfo = null，可以获取出需要分配的内存长度plLen
PLAYM4_API BOOL __stdcall PlayM4_GetStreamAdditionalInfo(LONG nPort, LONG lType, BYTE* pInfo, LONG* plLen);

///<图像处理
PLAYM4_API BOOL __stdcall PlayM4_SetColor(LONG nPort, DWORD nRegionNum, int nBrightness, int nContrast, int nSaturation, int nHue);
PLAYM4_API BOOL __stdcall PlayM4_GetColor(LONG nPort, DWORD nRegionNum, int *pBrightness, int *pContrast, int *pSaturation, int *pHue);
PLAYM4_API BOOL __stdcall PlayM4_SetImageSharpen(LONG nPort,DWORD nLevel);
PLAYM4_API BOOL __stdcall PlayM4_SetRotateAngle(LONG nPort, DWORD nRegionNum, DWORD dwType);

///<其他
PLAYM4_API DWORD __stdcall PlayM4_GetFileHeadLength();
PLAYM4_API DWORD __stdcall PlayM4_GetSdkVersion();
PLAYM4_API DWORD __stdcall PlayM4_GetLastError(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_SetSycGroup(LONG nPort, DWORD dwGroupIndex);
PLAYM4_API BOOL __stdcall PlayM4_MotionFlow(LONG nPort, DWORD dwAdjustType);

//私有信息回调接口,nStamp表示相对时间戳信息；pSyncData表示私有信息内容, nType表示需要回调的私有信息类型，支持多个私有信息类型一起设置，
//比如当nType等于6表示外部需要回调深眸目标框和普通规则框，此时播放库只会回调这两种类型，如果都需要回调则直接设置nType等于0即可，此时默认智能信息结构体里的信息都会回调；
//nMode表示回调注册的模式(用于接口的功能扩展)，0表示解码后立刻回调，播放库内部不进行和视频帧的匹配，也不会回调HDC，目前只支持0模式，后续根据需求可以增加匹配后回调模式和匹配后回调HDC模式
PLAYM4_API BOOL __stdcall PlayM4_RegisterIVSIntelInfoCB(LONG nPort, void (CALLBACK* IVSIntelInfoCBFun)(LONG nPort, HDC hDC, LONG nStamp, PRIVATE_INFO* pSyncData, void*  dwUser), void* dwUser, DWORD nType, DWORD nMode);


// 根据相对时间获取MP4封装在线定位偏移
PLAYM4_API int __stdcall PlayM4_GetMpOffset(int nPort, int nTime, int* nOffset);

//Hlog写日志开关接口。任一port的开关操作对该进程中所有port都会起作用。目前配置文件路径参数不起作用，作为后续可能的扩展功能预留。
PLAYM4_API BOOL __stdcall PlayM4_SetHLogFlag(int nPort, BOOL bFlag, char* pConfigFilePath);

#ifndef PLAYM4_HIKD3D11_TAG
#define PLAYM4_HIKD3D11_TAG
///<硬解码探测分辨率
#define HXVA_RESOLUTION_NONE   0x00      // 0
#define HXVA_RESOLUTION_200W   0x01      // 1920*1080
#define HXVA_RESOLUTION_300W   0x02      // 2048*1536
#define HXVA_RESOLUTION_500W   0x03      // 2560*1920
#define HXVA_RESOLUTION_800W   0x04      // 3840*2160
#define HXVA_RESOLUTION_1600W  0x05      // 4096*4096
#define HXVA_RESOLUTION_6400W  0x06      // 8192*8192(H.265)

///<硬解码和渲染能力集探测输出结构体
typedef struct _tagHDECODESUPPORT_
{
    unsigned char  chGPUType;                // GPU类型： 0-无，1-N卡独显，2-A卡独显，3-Intel核显
    unsigned char  bDXVA_D3D9;               // 0-不支持，1-支持
    unsigned char  bCUVID_D3D11;             // 0-不支持，1-支持
    unsigned char  chDXVAH264_Max_Resolution;// DXVA,h264编码支持的最大的分辨率大概值-不是精准值-具体定义见枚举类型
    unsigned char  chDXVAH265_Max_Resolution;// DXVA,h265编码支持的最大的分辨率大概值-不是精准值-具体定义见枚举类型
    unsigned char  chCUVIDH264_Max_Resolution;// CUVID,h264编码支持的最大的分辨率大概值-不是精准值-具体定义见枚举类型
    unsigned char  chCUVIDH265_Max_Resolution;// CUVID,h265编码支持的最大的分辨率大概值-不是精准值-具体定义见枚举类型
    unsigned char  chValidFlag;               ///<有效
    unsigned char  bD3D11VA;                  // 判断是否支持D3D11VA硬解码:0-不支持，1-支持
    unsigned char  chD3D11VAH264_Max_Resolution;// D3D11VA,h264编码支持的最大的分辨率大概值-不是精准值
    unsigned char  chD3D11VAH265_Max_Resolution;// D3D11VA,h265编码支持的最大的分辨率大概值-不是精准值
    unsigned char  nReserved[9];        // 保留字段
}HDECODESUPPORT;///< 硬解码支持信息

typedef struct _tagRENDERSUPPORT_
{
    unsigned char  bDDraw;              ///< 判断是否支持Draw渲染
    unsigned char  bD3D9Surface;        ///< 判断是否支持D3D9表面渲染
    unsigned char  bD3D9Texture;        ///< 判断是否支持D3D9纹理渲染
    unsigned char  bD3D11;              ///< 判断是否支持D3D11渲染
    unsigned char  chValidFlag;         ///< 有效
    unsigned char  nGPUType;            ///< 渲染探测GPU类型  0-无; 1-Intel;2-NVIDIA;3-AMD
    unsigned char  nReserved[10];       ///< 保留字段
}RENDERSUPPORT;///< 渲染支持信息

typedef struct _tagENGINESUPPORT_
{
    HDECODESUPPORT stHDecodeSupport; ///<硬解码能力集
    RENDERSUPPORT  stRenderSupport;  ///<渲染能力集
    unsigned int   chReserved[4];    ///<保留字段
}ENGINESUPPORT;///< 硬解码和渲染支持能力集

typedef struct _tagENGINESUPPORT_EX_
{
    HDECODESUPPORT stHDecodeSupportD3D9[8];
    HDECODESUPPORT stHDecodeSupportD3D11[8];
    RENDERSUPPORT  stRenderSupport[8];
    unsigned char  chD3D9DeviceCount;
    unsigned char  chD3D11DeviceCount;
    unsigned char  chRenderDeviceCount;
    unsigned char  chReserved[13];
}ENGINESUPPORT_EX;///< 硬解码和渲染支持能力集-EX

///<d3d11渲染抓图结构体
typedef struct _tagD3D11_PIC_INFO_
{
    unsigned int   nPicMode;     //抓图模式：0-按照码流实际分辨率抓图(之前抓图模式)；1-按照下面传入的宽高抓图（nPicWidth*nPicHeight为显示窗口宽高时效果最佳）
    unsigned char* pBuf;         //抓图数据buffer
    unsigned int   nBufSize;     //缓存buffer大小-（如果nPicMode为0则为之前的根据获取到的码流分辨率来进行缓存申请；如果nPicMode为1则上层根据设置的分辨率来申请缓存）
    unsigned int*  pPicSize;     //实际图片大小
    unsigned int   nPicWidth;    //设置抓图宽-nPicMode为1时有效，且宽>=32,nPicWidth*nPicHeight<5000w分辨率。
    unsigned int   nPicHeight;   //设置抓图高-nPicMode为1时有效，且高>=32,nPicWidth*nPicHeight<5000w分辨率。
    unsigned char  chReserve[8]; //reserve保留字段
}D3D_PIC_INFO;

/*D3D11图像后处理类型*/
typedef enum tagPLAYM4PostProcType
{
    PLAYM4_PPT_BRIGHTNESS               = 0x1,            ///< 亮度   [-1.0, 1.0]
    PLAYM4_PPT_HUE                      = 0x2,            ///< 色度   [0.0, 1.0]----0.166累加为一种颜色变化及0和1为同一种颜色
    PLAYM4_PPT_SATURATION               = 0x3,            ///< 饱和度 [-1.0, 1.0]
    PLAYM4_PPT_CONTRAST                 = 0x4,            ///< 对比度 [-1.0, 1.0]
    PLAYM4_PPT_SHARPNESS                = 0x5,            ///< 锐度   [ 0.0, 1.0]
}PLAYM4PostProcType;

#endif

///< 获取设备硬解码和渲染能力集--getport之后
PLAYM4_API int __stdcall PlayM4_GetEngineSupport(long nPort, ENGINESUPPORT* pstEngineSupport);

///<渲染库d3d11抓图-只支持d3d11渲染时有效，d3d9渲染无效:nType:0-jpeg,1-bmp.
PLAYM4_API int __stdcall PlayM4_GetD3DCapture(long nPort, unsigned int nType, D3D_PIC_INFO* pstPicInfo);


///<d3d11图像后处理-暂只支持d3d11渲染后处理-设置参数
PLAYM4_API int __stdcall PlayM4_SetD3DPostProcess(long nPort, PLAYM4PostProcType nPostType, float fValue);

///<d3d11图像后处理-暂只支持d3d11渲染后处理-获取参数
PLAYM4_API int __stdcall PlayM4_GetD3DPostProcess(long nPort, PLAYM4PostProcType nPostType, float* fValue);

///<设置D3D11渲染时(D3D9渲染时设置实际无效)字符叠加的字体库路径（不设置时采用默认字体库）-play前设置
PLAYM4_API int __stdcall PlayM4_SetConfigFontPath(int nPort, char* pFontPath);

////////////////////////////////////


///<SDP取流接口//////////////////////////////////////////////////////////////////////////
#ifndef PLAYM4_SESSION_INFO_TAG
#define PLAYM4_SESSION_INFO_TAG

//nProtocolType
#define PLAYM4_PROTOCOL_RTSP    1
//nSessionInfoType
#define PLAYM4_SESSION_INFO_SDP 1

typedef struct _PLAYM4_SESSION_INFO_   //交互信息结构
{
    int            nSessionInfoType;   //交互信息类型，暂时只支持SDP
    int            nSessionInfoLen;    //交互信息长度
    unsigned char* pSessionInfoData;   //交互信息数据
} PLAYM4_SESSION_INFO;
#endif

PLAYM4_API BOOL __stdcall PlayM4_OpenStreamAdvanced(LONG nPort, int nProtocolType, PLAYM4_SESSION_INFO* pstSessionInfo, DWORD nBufPoolSize);

///<鱼眼相关接口//////////////////////////////////////////////////////////////////////////
#ifndef FISH_EYE_TAG
#define FISH_EYE_TAG

// 安装类型
typedef enum tagFECPlaceType
{
    FEC_PLACE_WALL    = 0x1,        // 壁装方式     (法线水平)
    FEC_PLACE_FLOOR   = 0x2,        // 地面安装     (法线向上)
    FEC_PLACE_CEILING = 0x3,        // 顶装方式     (法线向下)

}FECPLACETYPE;

typedef enum tagFECCorrectType
{
    FEC_CORRECT_NULL                    = 0x0000, // 不矫正 - (鱼眼原图)
    FEC_CORRECT_PTZ                     = 0x0100, // PTZ
    FEC_CORRECT_180                     = 0x0200, // 180度矫正 - （对应2P） - 顶装/底装
    FEC_CORRECT_360                     = 0x0300, // 360全景矫正 - （对应1P） - 顶装/底装
    FEC_CORRECT_LAT                     = 0x0400, // 纬度展开 - 壁装
    FEC_CORRECT_SEMISPHERE              = 0x0500, // 鱼眼半球矫正 - (球面矫正)
    FEC_CORRECT_CYLINDER                = 0x0600, // 鱼眼圆柱显示 - 顶装/底装
    FEC_CORRECT_CYLINDER_SPLIT          = 0x0700, // 鱼眼切开的圆柱显示 - 顶装/底装
    FEC_CORRECT_PLANET                  = 0x0800, // 鱼眼小行星
    FEC_CORRECT_ARCSPHERE_HORIZONTAL    = 0x0900, // 鱼眼水平弧面 - 壁装
    FEC_CORRECT_ARCSPHERE_VERTICAL      = 0x0A00, // 鱼眼垂直弧面 - 壁装
}FECCORRECTTYPE;

typedef enum tagFECCorrectEffect
{
    FEC_CORRECT_EFFECT_BACK_FACE_CULLING    = 0x100,        // 背面剔除：非0表示启用，为0表示不启用，建议设置整数强转浮点数

}FECCORRECTEFFECT;

typedef struct tagCycleParam
{
    float   fRadiusLeft;    // 圆的最左边X坐标
    float   fRadiusRight;   // 圆的最右边X坐标
    float   fRadiusTop;     // 圆的最上边Y坐标
    float   fRadiusBottom;  // 圆的最下边Y坐标

}CYCLEPARAM;

typedef struct tagPTZParam
{
    float fPTZPositionX;        // PTZ 显示的中心位置 X坐标
    float fPTZPositionY;        // PTZ 显示的中心位置 Y坐标	

}PTZPARAM;

#define         FEC_UPDATE_RADIUS            0x1
#define         FEC_UPDATE_PTZZOOM           0x2
#define         FEC_UPDATE_WIDESCANOFFSET    0x4
#define         FEC_UPDATE_PTZPARAM          0x8
#define         FEC_UPDATT_PTZCOLOR          0x10
#define         FEC_UPDATE_PTZSELECT         0x20

// 色彩结构体
typedef struct tagFECColor
{
    unsigned char nR;     // R分量
    unsigned char nG;	  // G分量
    unsigned char nB;     // B分量
    unsigned char nAlpha; // Alpha分量
}FECCOLOR;

//PTZ框选结构体
typedef struct tagFECRectF
{
    float fTop;
    float fBottom;
    float fLeft;
    float fRight;
}FECRECTF;

typedef struct tagFECParam
{
    unsigned int    nUpDateType;            // 更新的类型
    unsigned int    nPlaceAndCorrect;       // 安装方式和矫正方式，只能用于获取，SetParam的时候无效,该值表示安装方式和矫正方式的和
    PTZPARAM        stPTZParam;             // PTZ 校正的参数
    CYCLEPARAM      stCycleParam;           // 鱼眼图像圆心参数
    float           fZoom;                  // PTZ 显示的范围参数
    float           fWideScanOffset;        // 180或者360度校正的偏移角度
    FECCOLOR        stPTZColor;             // PTZ颜色
    FECRECTF        stPTZSelct;             // PTZ选择区域
    unsigned int    nCut;                   //鱼眼原图是否进行裁剪以填充整个窗口（仅作用于鱼眼原图）
    int             nResver[10];            // 保留字段
}FISHEYEPARAM;

// PTZ在原始鱼眼图上轮廓的显示模式
typedef enum tagFECShowMode
{
    FEC_PTZ_OUTLINE_NULL,   // 不显示
    FEC_PTZ_OUTLINE_RECT,   // 矩形显示
    FEC_PTZ_OUTLINE_RANGE,  // 真实区域显示
}FECSHOWMODE;

#define    FEC_JPEG   0  // JPEG抓图
#define    FEC_BMP    1  // BMP 抓图

// 硬解码标志
#define FEC_DISPLAYSURFACE          0x400  // 鱼眼硬解标识

// 旋转单元结构体
typedef struct tagPLAYM4SRTransformElement
{
    float fAxisX;
    float fAxisY;
    float fAxisZ;
    float fValue; // 旋转的角度

}PLAYM4SRTRANSFERELEMENT;

// 旋转组合参数
typedef struct tagPLAYM4SRTransformParam
{
    PLAYM4SRTRANSFERELEMENT* pTransformElement;		// 旋转的坐标轴
    unsigned int		     nTransformCount;		// 旋转的组合次数
}PLAYM4SRTRANSFERPARAM;


//////////*************鱼眼相关的不建议使用接口和定义(老球面鱼眼)************************//////////////

#define FEC_DISPLAYSPHERE           0x402  // 鱼眼球面渲染显示(老球面鱼眼)-不建议使用

PLAYM4_API BOOL __stdcall PlayM4_FEC_Rotate(LONG nPort, PLAYM4SRTRANSFERPARAM *pstRotateParam);///<此接口为不建议使用接口(老球面鱼眼旋转接口)

//////////////////////**************************/////////////////////////


// 视角变换类型
typedef enum tagPLAYM4HRViewParamType
{
    PLAYM4_HR_VPT_ROTATION_X       = 0x1,          ///<水平旋转
    PLAYM4_HR_VPT_ROTATION_Y       = 0x2,          ///<垂直旋转
    PLAYM4_HR_VPT_SCALE            = 0x3,          ///<缩放(其绝对值为浮点型0值时为黑屏-正常效果)
}PLAYM4HRVIEWPARAMTYPE;

// 鱼眼3D模型参数
typedef enum tagPLAYM4FEC3DModelParam
{
    PLAYM4_FEC_3DMP_CYLINDER_HEIGHT              = 0x1,       ///< 圆柱模型高
    PLAYM4_FEC_3DMP_CYLINDER_RADIUS              = 0x2,       ///< 圆柱模型半径
}PLAYM4FEC3DMODELPARAM;

// 特定视角状态
typedef enum tagPLAYM4FECSpecialViewType
{
    PLAYM4_FEC_SVT_EDGE                          = 0x1        ///<弧面鱼眼模型与窗口贴边视角
}PLAYM4FECSPECIALVIEWTYPE;

typedef void (__stdcall* FISHEYE_CallBack )( void* pUser, unsigned int nSubPort, unsigned int nCBType, void* hDC, unsigned int nWidth, unsigned int nHeight);

#endif
// 启用鱼眼
PLAYM4_API BOOL __stdcall PlayM4_FEC_Enable(LONG nPort);

// 关闭鱼眼模块
PLAYM4_API BOOL __stdcall PlayM4_FEC_Disable(LONG nPort);

// 获取鱼眼矫正处理子端口 [1~31]
PLAYM4_API BOOL __stdcall PlayM4_FEC_GetPort(LONG nPort, unsigned int* nSubPort, FECPLACETYPE emPlaceType, FECCORRECTTYPE emCorrectType);

// 删除鱼眼矫正处理子端口
PLAYM4_API BOOL __stdcall PlayM4_FEC_DelPort(LONG nPort, unsigned int nSubPort);

// 设置鱼眼矫正参数
PLAYM4_API BOOL __stdcall PlayM4_FEC_SetParam(LONG nPort, unsigned int nSubPort, FISHEYEPARAM* pPara);

// 获取鱼眼矫正参数
PLAYM4_API BOOL __stdcall PlayM4_FEC_GetParam(LONG nPort, unsigned int nSubPort, FISHEYEPARAM* pPara);

// 设置显示窗口，可以随时切换
PLAYM4_API BOOL __stdcall PlayM4_FEC_SetWnd(LONG nPort, unsigned int nSubPort, void* hWnd);

// 设置鱼眼窗口的绘图回调
PLAYM4_API BOOL __stdcall PlayM4_FEC_SetCallBack(LONG nPort, unsigned int nSubPort, FISHEYE_CallBack cbFunc, void* pUser);

// 鱼眼抓图
PLAYM4_API BOOL __stdcall PlayM4_FEC_Capture(LONG nPort, unsigned int nSubPort, unsigned int nType, char* pFileName);

// 鱼眼硬解
PLAYM4_API BOOL __stdcall PlayM4_FEC_SetConfig(LONG nPort,unsigned int nType);

PLAYM4_API BOOL __stdcall PlayM4_FEC_GetCurrentPTZPort(LONG nPort, float fPositionX,float fPositionY, unsigned int *pnPort);

PLAYM4_API BOOL __stdcall PlayM4_FEC_SetCurrentPTZPort(LONG nPort, unsigned int nSubPort);

PLAYM4_API BOOL __stdcall PlayM4_FEC_SetPTZOutLineShowMode(LONG nPort,FECSHOWMODE nPTZShowMode);

//新的鱼眼半球矫正视角变化(旋转)两个相关接口
//获取半球矫正视角参数(设置前先获取当前的值)
PLAYM4_API BOOL __stdcall PlayM4_FEC_GetViewParam(LONG nPort, unsigned int nSubPort, PLAYM4HRVIEWPARAMTYPE enViewParamType, float* fValue);
//设置半球矫正视角变化参数(设置的值为获取值加上要偏移值)
PLAYM4_API BOOL __stdcall PlayM4_FEC_SetViewParam(LONG nPort, unsigned int nSubPort, PLAYM4HRVIEWPARAMTYPE enViewParamType, float fValue);

//鱼眼电子放大 nType = 0，
//鱼眼窗口分割 nType = 1，
//注意，窗口分割时hDestWnd无效，请设置为NULL（20180813暂不支持）
//目前电子放大不支持其他窗口上放大，nRegionNum置为0，hDestWnd置为NULL，bEnable置为0取消电子放大，非0为电子放大
//pSrcRect区域归一化，乘以1000除以宽或者高（其值在0-1000之间）
//只推荐原图、180、360、纬度展开进行电子放大-3D鱼眼不建议使用此接口(用视角变化接口进行缩放），ptz和3D矫正方式不支持电子放大
PLAYM4_API BOOL __stdcall PlayM4_FEC_SetDisplayRegion(LONG nPort, unsigned int nSubPort,unsigned int nType, DWORD nRegionNum, RECT *pSrcRect, HWND hDestWnd, BOOL bEnable);
//背面剔除：非0表示启用，为0表示不启用，建议设置整数强转浮点数
PLAYM4_API BOOL __stdcall PlayM4_FEC_SetCorrectEffect(LONG nPort, unsigned int nSubPort, FECCORRECTEFFECT nCorrectEffect, float fValue);

// 设置3D模型参数-针对3D圆柱展开有效
PLAYM4_API BOOL __stdcall PlayM4_FEC_Set3DModelParam(int nPort, unsigned int nSubPort, PLAYM4FEC3DMODELPARAM enType, float fValue);

// 获取特定视角参数 - 只适用于弧面，跟SetViewParam接口配合使用
PLAYM4_API BOOL __stdcall PlayM4_FEC_GetSpecialViewParam(int nPort, unsigned int nSubPort, PLAYM4FECSPECIALVIEWTYPE enSVType, PLAYM4HRVIEWPARAMTYPE enVPType, float* pValue);

///<图像增强相关接口//////////////////////////////////////////////////////////////////////////
#ifndef PLAYM4_HIKVIE_TAG
#define PLAYM4_HIKVIE_TAG

typedef struct _PLAYM4_VIE_DYNPARAM_
{
    int moduFlag;      //启用的算法处理模块，在PLAYM4_VIE_MODULES中定义，如 PLAYM4_VIE_MODU_ADJ | PLAYM4_VIE_MODU_EHAN，模块启用后，必须设置相应的参数；
    //PLAYM4_VIE_MODU_ADJ
    int brightVal;     //亮度调节值，[-255, 255]
    int contrastVal;   //对比度调节值，[-256, 255]
    int colorVal;      //饱和度调节值，[-256, 255]
    //PLAYM4_VIE_MODU_EHAN
    int toneScale;     //滤波范围，[0, 100]
    int toneGain;      //对比度调节，全局对比度增益值，[-256, 255]
    int toneOffset;    //亮度调节，亮度平均值偏移，[-255, 255]
    int toneColor;     //颜色调节，颜色保真值，[-256, 255]
    //PLAYM4_VIE_MODU_DEHAZE
    int dehazeLevel;   //去雾强度，[0, 255]
    int dehazeTrans;   //透射值，[0, 255]
    int dehazeBright;  //亮度补偿，[0, 255]
    //PLAYM4_VIE_MODU_DENOISE
    int denoiseLevel;  //去噪强度，[0, 255]
    //PLAYM4_VIE_MODU_SHARPEN
    int usmAmount;     //锐化强度，[0, 255]
    int usmRadius;     //锐化半径，[1, 15]
    int usmThreshold;  //锐化阈值，[0, 255]
    //PLAYM4_VIE_MODU_DEBLOCK
    int deblockLevel;  //去块强度，[0, 100]
    //PLAYM4_VIE_MODU_LENS
    int lensWarp;      //畸变量，[-256, 255]
    int lensZoom;      //缩放量，[-256, 255]
    //PLAYM4_VIE_MODU_CRB
    //无响应参数
} PLAYM4_VIE_PARACONFIG;

typedef enum _PLAYM4_VIE_MODULES
{
    PLAYM4_VIE_MODU_ADJ      = 0x00000001, //图像基本调节
    PLAYM4_VIE_MODU_EHAN     = 0x00000002, //局部增强模块
    PLAYM4_VIE_MODU_DEHAZE   = 0x00000004, //去雾模块
    PLAYM4_VIE_MODU_DENOISE  = 0x00000008, //去噪模块
    PLAYM4_VIE_MODU_SHARPEN  = 0x00000010, //锐化模块
    PLAYM4_VIE_MODU_DEBLOCK  = 0x00000020, //去块滤波模块
    PLAYM4_VIE_MODU_CRB      = 0x00000040, //色彩平衡模块
    PLAYM4_VIE_MODU_LENS     = 0x00000080, //镜头畸变矫正模块
}PLAYM4_VIE_MODULES;
#endif

//设置关闭/开启模块
//dwModuFlag对应PLAYM4_VIE_MODULES宏,可组合
//先设置模块开启，再设置模块参数；期间采用默认的参数;
//关闭模块后，上次设置的参数清空
//其他接口调用，必须在该接口开启模块后；否则，返回错误
PLAYM4_API BOOL __stdcall PlayM4_VIE_SetModuConfig(LONG lPort,int nModuFlag,BOOL bEnable);

//设置图像增强区域，NULL全图；超过全图，采用全图；最小区域32*32像素
//可支持设置区域，最多比较说4个，第一个版本可以只支持一个。多个区域要求不能重叠，有重叠就报错
PLAYM4_API BOOL __stdcall PlayM4_VIE_SetRegion(LONG lPort,LONG lRegNum,RECT* pRect);

//获取开启模块
PLAYM4_API BOOL __stdcall PlayM4_VIE_GetModuConfig(LONG lPort,int* pdwModuFlag);

//设置参数，未开启模块的参数设置被忽略
PLAYM4_API BOOL __stdcall PlayM4_VIE_SetParaConfig(LONG lPort,PLAYM4_VIE_PARACONFIG* pParaConfig);

//获取开启模块的参数
PLAYM4_API BOOL __stdcall PlayM4_VIE_GetParaConfig(LONG lPort,PLAYM4_VIE_PARACONFIG* pParaConfig);

typedef struct
{
    long    lDataType;          //私有数据类型
    long    lDataStrVersion;    //数据返回的结构体版本，主要是为了兼容性
    long    lDataTimeStamp;
    long    lDataLength;
    char*   pData;
}AdditionDataInfo;

PLAYM4_API BOOL __stdcall PlayM4_SetAdditionDataCallBack(LONG nPort, DWORD nSyncType, void (CALLBACK* AdditionDataCBFun)(long nPort, AdditionDataInfo* pstAddDataInfo, void* pUser), void* pUser);


#define SOFT_DECODE_ENGINE 0 ///<软解码
#define HARD_DECODE_ENGINE 1 ///<硬解码

PLAYM4_API BOOL  __stdcall PlayM4_SetDecodeEngineEx(LONG nPort, DWORD nDecodeEngine);
PLAYM4_API DWORD __stdcall PlayM4_GetDecodeEngine(LONG nPort);

#define PLAYM4_SOURCE_MODULE             0 // 数据源模块
#define PLAYM4_DEMUX_MODULE              1 // 解析模块
#define PLAYM4_DECODE_MODULE             2 // 解码模块
#define PLAYM4_RENDER_MODULE             3 // 渲染模块

#define PLAYM4_RTINFO_HARDDECODE_ERROR   0 // 硬解码致命(需要切软解)错误
#define PLAYM4_RTINFO_SOFTDECODE_ERROR   1 // 软解码错误(暂无)
#define PLAYM4_RTINFO_MEDIAHEADER_ERROR  2 // 媒体头错误
#define PLAYM4_RTINFO_SWITCH_SOFT_DEC    3 // 切换至软解
#define PLAYM4_RTINFO_ALLOC_MEMORY_ERROR 4 // 内存分配失败
#define PLAYM4_RTINFO_ENCRYPT_ERROR      5 // 秘钥错误 [加密的码流在解析报错时也会回调该错误码]
#define PLAYM4_RTINFO_RENDER_OVER        8 // 渲染一帧结束
#define PLAYM4_RTINFO_ERR_PRESENT             16 // 渲染显示错误[当前渲染引擎渲染失败,上层重新切换引擎]
#define PLAYM4_RTINFO_IDMX_DATA_ERROR         32 // 数据有误,解析失败
#define PLAYM4_RTINFO_DECODE_PARAM_ERROR      64 // 参数有误,解码失败
#define PLAYM4_RTINFO_SOFTDECODE_DATA_ERROR   128 // 软解码数据错误
#define PLAYM4_RTINFO_IDMX_PSH_PSM_ERROR         0x100// PSH/PSM有误,解析失败
#define PLAYM4_RTINFO_IDMX_RTP_HEADER_ERROR      0x200// RTP头有误，包括扩展头
#define PLAYM4_RTINFO_IDMX_RTP_HEADER_SEQ_ERROR  0x400// RTP视频包序不连续
#define PLAYM4_RTINFO_IDMX_REDUNDANT_ERROR       0x800// 码流里有冗余数据
#define PLAYM4_RTINFO_IDMX_MEDIA_CHANGE_ERROR             0x1000// 媒体信息发生改变(与hik头的信息不一致或者psm里面的音视频类型发生改变)
#define PLAYM4_RTINFO_IDMX_DIFFERENT_FRAMERATE_ERROR      0x2000// 封装层帧率与编码层帧率不一致
#define PLAYM4_RTINFO_IDMX_DIFFERENT_RESOLUTION_ERROR     0x4000// 封装层分辨率与编码层分辨率不一致
#define PLAYM4_RTINFO_IDMX_DECORD_ERROR                   0x8000// 编码检查异常
#define PLAYM4_RTINFO_IDMX_PRIVT_LEN_ERROR                0x10000// 私有数据长度异常
#define PLAYM4_RTINFO_SOFTDECODE_DATA_FLAKE_ERROR         0x20000 //软解码数据存在花屏错误

typedef struct
{
    int            nRunTimeModule;     //当前运行模块
    int            nStrVersion;        //结构体版本
    int            nFrameTimeStamp;    //帧号
    int            nFrameNum;          //时间戳
    int            nErrorCode;         //状态码-并非都是错误信息
    unsigned char  reserved[12];       //保留字节
}RunTimeInfo;

// 要求不同模块的回调函数必须为同一个，否则后注册的将覆盖掉先注册的
PLAYM4_API BOOL __stdcall PlayM4_SetRunTimeInfoCallBackEx(LONG nPort, int nModule, void (CALLBACK* RunTimeInfoCBFun)(long nPort, RunTimeInfo* pstRunTimeInfo, void* pUser), void* pUser);

// 1、SetRunTimeInfoCallBackEx接口，nErrorCode类型6改为类型8；以后扩展方式 16、32、64，且从8开始消息默认关闭不发送。
// 2、新增实时消息控制接口，控制从8以后的消息，支持从8开始后消息，拼接模式 8|16|32 这种方式，上层主动选择接受类型。
// 3、nType可以通过或的方式将类型传入，nFlag表示清除（为0）或者进行回调（非0）；
PLAYM4_API int __stdcall PlayM4_SetRunTimeInfoCallbackType(int nPort, int nModule, unsigned int nType, int nFlag);


#ifndef PLAYM4_HIKSR_TAG
#define PLAYM4_HIKSR_TAG

#define CFG_DISPLAY_MODEL_MODE         0x499

// 模型贴图类型 0 双面 1 外面 2 内面 
typedef enum tagPLAYM4SRTextureMode
{
    PLAYM4_TEXTURE_DOUBLE       = 0x0,           // 双面贴纹理
    PLAYM4_TEXTURE_OUTER        = 0x1,           // 外表面贴纹理
    PLAYM4_TEXTURE_INNER        = 0x2,           // 内表面贴纹理
}PLAYM4SRTEXTUREMODE;

//模型类型
typedef enum tagPLAYM4SRModelType
{
    PLAYM4_MODELTYPE_HEMISPHERE = 0x0,           //鱼眼视频用半球模型
    PLAYM4_MODELTYPE_EAGLE_EYE  = 0x1,           //碗状类半球模型用于鹰眼拼接
    PLAYM4_MODELTYPE_CUBE       = 0x2,           //立方体模型
}PLAYM4SRMODELTYPE;

#define PLAYM4_MODEL_SOLID      0x0001           //实体模型
#define PLAYM4_MODEL_FRAME      0x0010           //边框模型
#define PLAYM4_MODEL_AXIS       0x0100           //坐标轴

//渲染到3D模型的相关参数
typedef struct tagRenderModelMode
{
    bool                          bModelMode;      // 是否将渲染到3D模型上
    PLAYM4SRTEXTUREMODE           emTextureMode;   // 纹理贴图模式 双面 外表面 内表面
    unsigned long                 ulDisplayType;   // 按位组合 SR_MODEL_SOLID | SR_MODEL_FRAME | SR_MODEL_AXIS
    unsigned int                  nTransformMode;  // 0 线框坐标轴随模型变换  1 线框坐标轴不随模型变换
    PLAYM4SRMODELTYPE             emModelType;     // 模型类型，鱼眼半球，鹰眼碗模型等
}PLAYM4SRMODELMODE;

#define SR_JPEG    0
#define SR_BMP     1

#endif


PLAYM4_API BOOL __stdcall PlayM4_SR_SetConfig(LONG nPort, int nCfgType, void *pValue);

PLAYM4_API BOOL __stdcall PlayM4_SR_Rotate(LONG nPort, PLAYM4SRTRANSFERPARAM *pstRotateParam);

PLAYM4_API BOOL __stdcall PlayM4_SR_Capture(LONG nPort, unsigned int nType, char *pFileName);



///<预录像相关接口//////////////////////////////////////////////////////////////////////////
#define PLAYM4_MEDIA_HEAD     1   //系统头数据
#define PLAYM4_VIDEO_DATA     2   //视频流数据
#define PLAYM4_AUDIO_DATA     3   //音频流数据
#define PLAYM4_PRIVT_DATA     4   //私有流数据

//预录像数据信息
typedef struct  
{
    long nType;                     // 数据类型，如文件头，视频，音频，私有数据等
    long nStamp;                    // 时间戳
    long nFrameNum;                 // 帧号
    long nBufLen;                   // 数据长度
    char* pBuf;                     // 帧数据，以帧为单位回调
    PLAYM4_SYSTEM_TIME  stSysTime;  // 全局时间
}RECORD_DATA_INFO;

/////////////////////////////////////////////////////////////////////////////////////////////
//Unproposed Interface 不建议使用的接口
/////////////////////////////////////////////////////////////////////////////////////////////
PLAYM4_API BOOL __stdcall PlayM4_RigisterDrawFun(LONG nPort,void (CALLBACK* DrawFun)(long nPort,HDC hDc,void* nUser),void* nUser);
PLAYM4_API BOOL __stdcall PlayM4_SetDecCallBack(LONG nPort,void (CALLBACK* DecCBFun)(long nPort,char * pBuf,long nSize,FRAME_INFO * pFrameInfo, void* nReserved1,void* nReserved2));
PLAYM4_API BOOL __stdcall PlayM4_SetDecCallBackEx(LONG nPort,void (CALLBACK* DecCBFun)(long nPort,char * pBuf,long nSize,FRAME_INFO * pFrameInfo, void* nReserved1,void* nReserved2), char* pDest, long nDestSize);
PLAYM4_API BOOL __stdcall PlayM4_SetSycStartTime(LONG nPort, PLAYM4_SYSTEM_TIME *pstSystemTime);
PLAYM4_API BOOL __stdcall PlayM4_SyncToAudio(LONG nPort, BOOL bSyncToAudio);
PLAYM4_API BOOL __stdcall PlayM4_SetDecodeEngine(LONG nPort, DWORD nDecodeEngine);
PLAYM4_API BOOL __stdcall PlayM4_SetRunTimeInfoCallBack(LONG nPort, void (CALLBACK* RunTimeInfoCBFun)(long nPort, RunTimeInfo* pstRunTimeInfo, void* pUser), void* pUser);

/////////////////////////////////////////////////////////////////////////////////////////////
//Unsupported Interface 不再支持的接口
/////////////////////////////////////////////////////////////////////////////////////////////
//设置预录像开关，bFlag=1开启，bFlag=0关闭
PLAYM4_API BOOL __stdcall PlayM4_SetPreRecordFlag(LONG lPort, BOOL bFlag);
//预录像数据数据回调
PLAYM4_API BOOL __stdcall PlayM4_SetPreRecordCallBack(LONG lPort, void (CALLBACK* PreRecordCBfun)(long nPort, RECORD_DATA_INFO* pRecordDataInfo, void* pUser), void* pUser);
PLAYM4_API BOOL __stdcall PlayM4_InitDDraw(HWND hWnd);
PLAYM4_API BOOL __stdcall PlayM4_RealeseDDraw();
PLAYM4_API BOOL __stdcall PlayM4_SetFileEndMsg(LONG nPort,HWND hWnd,UINT nMsg);
PLAYM4_API int  __stdcall PlayM4_GetCaps();
PLAYM4_API BOOL __stdcall PlayM4_SetOverlayMode(LONG nPort,BOOL bOverlay,COLORREF colorKey);
PLAYM4_API LONG __stdcall PlayM4_GetOverlayMode(LONG nPort);
PLAYM4_API COLORREF __stdcall PlayM4_GetColorKey(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_InitDDrawDevice();
PLAYM4_API void __stdcall PlayM4_ReleaseDDrawDevice();
PLAYM4_API DWORD __stdcall PlayM4_GetDDrawDeviceTotalNums();
PLAYM4_API BOOL __stdcall PlayM4_SetDDrawDevice(LONG nPort,DWORD nDeviceNum);
PLAYM4_API BOOL __stdcall PlayM4_GetDDrawDeviceInfo(DWORD nDeviceNum,LPSTR  lpDriverDescription,DWORD nDespLen,LPSTR lpDriverName ,DWORD nNameLen,HMONITOR *hhMonitor);
PLAYM4_API int  __stdcall PlayM4_GetCapsEx(DWORD nDDrawDeviceNum);
PLAYM4_API BOOL __stdcall PlayM4_SetDDrawDeviceEx(LONG nPort,DWORD nRegionNum,DWORD nDeviceNum);
PLAYM4_API BOOL __stdcall PlayM4_OpenStreamEx(LONG nPort,PBYTE pFileHeadBuf,DWORD nSize,DWORD nBufPoolSize);
PLAYM4_API BOOL __stdcall PlayM4_CloseStreamEx(LONG nPort);
PLAYM4_API BOOL __stdcall PlayM4_InputVideoData(LONG nPort,PBYTE pBuf,DWORD nSize);
PLAYM4_API BOOL __stdcall PlayM4_InputAudioData(LONG nPort,PBYTE pBuf,DWORD nSize);
PLAYM4_API BOOL __stdcall PlayM4_SetTimerType(LONG nPort,DWORD nTimerType,DWORD nReserved);
PLAYM4_API BOOL __stdcall PlayM4_GetTimerType(LONG nPort,DWORD *pTimerType,DWORD *pReserved);
PLAYM4_API BOOL __stdcall PlayM4_SetVerifyCallBack(LONG nPort, DWORD nBeginTime, DWORD nEndTime, void (__stdcall * funVerify)(long nPort, FRAME_POS * pFilePos, DWORD bIsVideo, DWORD nUser),  DWORD  nUser);
PLAYM4_API BOOL __stdcall PlayM4_SetAudioCallBack(LONG nPort, void (__stdcall * funAudio)(long nPort, char * pAudioBuf, long nSize, long nStamp, long nType, long nUser), long nUser);
PLAYM4_API BOOL __stdcall PlayM4_SetEncChangeMsg(LONG nPort,HWND hWnd,UINT nMsg);
PLAYM4_API BOOL __stdcall PlayM4_GetOriginalFrameCallBack(LONG nPort, BOOL bIsChange,BOOL bNormalSpeed,long nStartFrameNum,long nStartStamp,long nFileHeader,void(CALLBACK *funGetOrignalFrame)(long nPort,FRAME_TYPE *frameType, long nUser),long nUser);
PLAYM4_API BOOL __stdcall PlayM4_GetFileSpecialAttr(LONG nPort, DWORD *pTimeStamp,DWORD *pFileNum ,DWORD *pReserved);
PLAYM4_API BOOL __stdcall PlayM4_SetPlayMode(LONG nPort,BOOL bNormal);
PLAYM4_API BOOL __stdcall PlayM4_SetOverlayFlipMode(LONG nPort,BOOL bTrue);
PLAYM4_API BOOL __stdcall PlayM4_SetDeflash(LONG nPort,BOOL bDefalsh);
PLAYM4_API BOOL __stdcall PlayM4_SetPicQuality(LONG nPort,BOOL bHighQuality);
PLAYM4_API BOOL __stdcall PlayM4_GetPictureQuality(LONG nPort,BOOL *bHighQuality);
/////////////////////////////////////////////////////////////////////////////////////////////

PLAYM4_API BOOL __stdcall PlayM4_SetGlobalBaseTime(long nPort, PLAYM4_SYSTEM_TIME stGlobalBaseTime);

#endif //_PLAYM4_H_
