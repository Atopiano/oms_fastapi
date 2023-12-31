import requests
from bs4 import BeautifulSoup
class ITGlossary:
    def __init__(self):
        self.glossary = {"atlassian":"아틀라시안", "aws":"아마존웹서비스", "gitlab":"깃랩", "cicd":"씨아이씨디", "ai":"인공지능", "api":"에이피아이", "sw":"소프트웨어", "nosql":"노에스큐엘", "django":"장고", "spring": "스프링", "boot": "부트", "engineer":"엔지니어"
                    , "product":"제품", "typescript":"타입스크립트", "nestjs":"네스트제이에스","nest.js":"넥스트제이에스", "python":"파이썬", "git":"깃", "kotlin":"코틀린", "mariadb":"마리아데이터베이스", "redis":"리디스", "frontend":"프론트엔드", "react":"리엑트", "openapi":"오픈에이피아이", "selenium":"셀레니움"
                    , "tool":"툴", "oracle":"오라클", "server":"서버", "kafka":"카프카", "github":"깃허브", "server":"서버","frontend":"프론트엔드","DBA":"데이터베이스엔지니어", "pm":"프로젝트매니저","fullstack":"풀스택", "web":"웹", "mobile": "모바일",
                    'devops':"시스템엔지니어", "java":"자바", "javascript":"자바스크립트","jquery":"제이쿼리","postgresql":"포스트그레에스큐엘","mysql":"마이에스큐엘","restapi":"레스트에이피아이","linux":"리눅스","node.js":"노드제이에스","c++":"씨플플","reactnative":"리액트네이티브"
                    ,"firebase":"파이어베이스","springdatajpa":"스프링데이터제이피에이","git":"깃","springboot":"스프링부트", "awsec2":"아마존일래스틱컴퓨트","jenkins":"젠킨스","swift":"스위프트","rubyonrails":"루비온레일즈","android":"안드로이드","php":"피에이치피","vue.js":"뷰제이에스","vuejs":"뷰제이에스"
                    ,"html5":"에이치티엠엘오","css3":"씨에스에스3","reactjs":"리엑트제이에스","apolloclient":"아폴로클라이언트","ubuntu":"우분투","babel":"바벨","webpack":'웹펙',"sass":"사스","graphql":"그래프큐엘","nextjs":"넥스트제이에스","next.js":"넥스트제이에스","typeorm":"타이프오알엠","expressjs":"익스프레스제이에스"
                    ,"apolloserver":"아폴로서버","html":"에이치티엠엘","css":"씨에스에스","styled-components":"스타일컴퍼넌트","storybook":"스토리북","docker":"도커","awsrds":"아마존관계형데이터베이스","ios":"아이오에스","objectivec":"오브젝티브씨","objective-c":"오브젝티브씨","flask":"플라스크","uwsgi":"유더블유에스지아이"
                    ,"nginx":"엔진엑스","awss3":"아마존스토리지삼","sql":"에스큐엘","awsiam":"아마존아이엠","awsvpc":"아마존브이피씨","terraform":"테라폼","angular":"앵귤러","redux.js":"리덕스","vuex":"뷰엑스","jpa":"제이피에이","junit":"제이유닛","rabbitmq":"래빗엠큐"
                    ,"kubernetes":"쿠버네티스","gradle":"그래들","svelte":"스벨트","angularjs":"앵귤러제이에스","yarn":"얀","es6":"이에스6","apache":"아파치","codeigniter":"코드이그나이터","rxswift":"알엑스스위프트","mongodb":"몽고데이터베이스","dba":"데이터베이스관리자","shell":"셸"
                    ,"auroradb":"오로라데이터베이스","babylon.js":"바빌론제이에스","solidity":"솔리디티","jira":"지라","truffle":"트러플","mssql":"엠에스에스큐엘","c":"씨","unity":"유니티","unrealengine":"언리얼엔진","directx":"다이렉트엑스","apachehadoop":"아파치하둡"
                    ,"apachekafka":"아파치카프카","embeddedsystem":"임베디드시스템","rtos":"알티오에스","redux-saga":"리덕스스사가","reduxsaga":"리덕스스사가","apachesentry":"아파치센트리","keras":"케라스","web3js":"웹제이에스","web3.js":"웹제이에스","ethersjs":"이덜스제이에스","ethers.js":"이덜스제이에스","rust":"러스트","vr":"브이알","vuetify.js":"뷰티파이제이에스"
                    ,"vuecli":"뷰클리","awswebservices":"아마존웹서비스","googlecloudplatform":"구글클라우드플랫폼","elasticsearch":"엘라스틱서치","rancher":"랜처","gin":"진","tensorflow":"텐서플로","pytorch":"파이토치","awsecs":"아마존이씨에스","tableau":"태블로","r":"알"
                    ,"apachejmeter":"아파치제이미터","apachespark":"아파치스파크","scikit-learn":"사이킷런","scikitlearn":"사이킷런","pandas":"판다스","awscloudwatch":"아마존클라우드와치","apacheairflow":"아파치에어플로우","flutter":"플러터","threejs":"쓰리제이에스","bitbucket":"비트버킷"
                    ,"nuxt.js":"넉스트제이에스","nuxtjs":"넉스트제이에스","laravel":"라라벨","ci/cd":"씨아이씨디","cicd":"씨아이씨디","seo":"검색엔진최적화","awscodecommit":"아마존코드커밋","cleanarchitecture":"클린아키텍쳐","fastapi":"패스트에이피아이","drf":"디알에프","delphi":"델파이"
                    ,"opencv":"오픈씨브이","unityml-agents":"유니티머신러닝에이전트","unitymlagents":"유니티머신러닝에이전트","windows":"윈도우","scipy":"싸이피","numpy":"넘파이","zeplin":"제플린","wpf":"더블유피에프","visualbasic":"비주얼베이식",".net":"닷넷","awscloud9":"아마존클라우드나인","blockchain":"블록체인","heroku":"헤로쿠"
                    ,"apachehive":"아파치하이브","tcpip":"티씨피아이피","tcp":"티씨피","ip":"아이피","centos":"센토스","azure":"애저","shellscript":"셸스크립트","ethereum":"이더리움","deeplearning":"딥러닝","machinevision":"머신비전","imageprocessing":"이미지프로세싱"
                    ,"microservicearchitecture":"마이크로서비스아키텍쳐","electron":"일렉트론","jest":"제스트","awsalexa":"아마존알렉사","reactadmin":"리액트어드민","ajax":"에이잭스","jsp":"제이에스피","circleci":"써클씨아이","springsecurity":"스프링시큐리티","ros":"알오에스"
                    ,"qt":"큐티","newrelic":"뉴렐릭","mxnet":"엠엑스넷","nvidiatensorrt":"엔비디아텐서알티","tensorflowlite":"텐서플로라이트","bootstrap":"부트스트랩","nlp":"엔엘피","mvc":"엠브이씨","googleappengine":'구글앱엔진',"apachehttpserver":"아파치에이치티티피서버"
                    ,"awscloudfront":"아마존클라우드프론트","matlab":"매트랩","cuda":"쿠다","opencl":"오픈씨엘","labview":"랩뷰","eslint":"이에스린트","websocket":"웹소켓","mybatis":"마이바티스","catia":"카티아","awsbackup":"아마존백업","dart":"다트","sqlite":"에스큐엘라이트"
                    ,"awslambda":"아마존람다","mvvm":"엠브이브이엠","mvp":"엠브이피","canvas":"캔버스","hibernate":"하이버네이트","retrofit":"레트로피트","mobx":"몹엑스","awseks":"아마존이케이에스","assembly":"어셈블리","xaml":"엑스에이엠엘","springcloud":"스프링클라우드"
                    ,"apachezookeeper":"아파치주키퍼","es5":"이에스5","mantisbugtracker":"맨티스버그트래커","klaytn":"클레이튼","rpc":"알피씨","awswafshield":"아마존쉴드","awsiotdevicedefender":"아이오티보안관리","azuremonitor":"애저모니터","azuresentinel":"애저센티넬"
                    ,"awscontroltower":"아마존컨트롤타워","awssecurityhub":"아마존시큐리티허브","azurevpngateway":"애저브이피엔게이트웨이","neo4j":"네오사제이","socketio":"소켓아이오","grpc":"지알피씨","sequelizejs":"시큐라이즈제이에스","mongoose":"몽구스","querydsl":"쿼리디에스엘"
                    ,"ansible":"앤서블","azurekubernetesservice":"애저쿠버네티스서비스","datadog":"데이터독","openssl":"오픈에스에스엘","linuxkernel":"리눅스커널","mfc":"엠에프씨","unix":"유닉스","awsdynamodb":"아마존다이나모데이터베이스","springbatch":"스프링배치","ffmpeg":"에프에프엠페그"
                    ,"gstreamer":"지에스스트리머","webrtc":"웹알티씨","cocos2d":"코코스투디","facebookapi":"페이스북에이피아이","pyqt":"피와이큐티","windowsembedded":"윈도우임베디드","bigdata":"빅데이터","iot":"아이오티","mqtt":"엠큐티티","slackapi":"슬랙에이피아이"
                    ,"githubactions":"깃허브액션","tailwindcss":"테일윈드씨에스에스","machinelearning":"머신러닝","reactor":"리액터","awsamplify":"아마존앰플리파이","highcharts":"하이차트","materialui":"매터리얼유아이","realm":"렘","cocoapods":"코코아팟","scala":"스칼라"
                    ,"hyperledger":"하이퍼레저","apachesolr":"아파치솔라","d3js":"디삼제이에스","jstl":"제이에스티엘","auth0":"어쓰제로","composer":"컴포저","protocolbuffers":"프로토콜버퍼","elixir":"엘릭서","awscodepipeline":"아마존코드파이프라인","webgl":"웹지엘"
                    ,"opengl":"오픈지엘","awscloudformation":"아마존클라우드포메이션","navercloudplatform":"네이버클라우드플랫폼","naver":"네이버","cad":"캐드","jsx":"제이에스엑스","reacthooksapi":"리액트훅스에이피아이","googlebigquery":"구글빅쿼리","awssagemaker":"아마존세이지메이커"
                    ,"awselasticbeanstalk":"아마존엘라스틱빈스토크","awssimplequeueservice":"아마존심플큐서비스","celery":"셀러리","vtk":"브이툴킷","openmp":"오픈엠피","sap":"에스에이피","crossbrowsing":"크로스브라우징","apacheflink":"아파치플링크","ar":"증강현실","mr":"혼합현실"
                    ,"web3py":"웹삼점파이","awsemr":"아마존이엠알","awsglue":"아마존글루","awsredshift":"아마존레드시프트","awsquicksight":"아마존퀵사이트","responsiveweb":"레스폰시브웹","pki":"피케이아이","vertx":"버텍스","database":"데이터베이스","dataanalysis":"데이터분석"
                    ,"elementui":"엘리멘트유아이","ble":"비엘이","splunk":"스플렁크","devexpress":"디벨롭익스프레스","retrofit2":"레트로핏","microsoftazure":"마이크로소프트애저","springdata":"스프링데이터","awsmq":"아마존엠큐","mlkit":"엠엘키트","oop":"객체지향프로그래밍"
                    ,"arm":"에이알엠","ida":"아이다","cypress":"사이프레스","presto":"프레스토","cnn":"씨엔엔","aspnet":"에이에스피넷","hlsl":"에이치엘에스엘","slam":"슬램","awsathena":"아마존아테나","apachenifi":"아파치나이파이","apacheoozie":"아파치우지","kendoui":"켄도유아이"
                    ,"qml":"큐엠엘","mocha":"모차","ruby":"루비","azuresqldatawarehouse":"애저에스큐엘데이터웨어하우스","nfc":"엔에프씨","codepush":"코드푸시","openstack":"오픈스택","redhatcephstorage":"레드햇세프스토리지","xamarin":"자마린","azurecomputervision":"애저컴퓨터비전"
                    ,"prisma":"프리스마","awsroute53":"아마존루트오삼","awscertificatemanager":"아마존인증서관리자","nexacro":"넥사크로","awssimplenotificationservice":"아마존심플노티피케이션서비스","torch":"토치","echarts":"이차트","googleanalytics":"구글분석","raspberrypi":"라즈베리파이"
                    ,"rxjava":"알엑스자바","memcached":"멤캐시디","puppeteer":"퍼피티어","reduxthunk":"리덕스썬크","awscodebuild":"아마존코드빌드","dlib":"디립","verilog":"베릴로그","googlecloudtoolsforintellij":"지씨티인텔리제이","yolo":"욜로","jni":"자바네이티브인터페이스"
                    ,"apachetomcat":"아파치톰캣","awsapigateway":"아마존에이피아이게이트웨이","awscodesstar":"아마존코드스타","awscodedeploy":"아마존코드디플로이","apachehbase":"아파치에이치베이스","windowsforms":"윈도우폼","gan":"지에이엔","svn":"에브이엔","autosar":"오토사"
                    ,"uml":"유엠엘","3dsmax":"3디에스맥스","sdn":"에스디엔","openflow":"오픈플로우","windowskernel":"윈도우커널","appium":"앱피움","cucumber":"큐컴버","ejs":"이제이에스","azurepowerbiembedded":"애저파워비아이임베디드","spinnaker":"스피내커","consul":"콘술"
                    ,"haproxy":"하프록시","netty":"네티","windowsserver":"윈도우서버","iocp":"아이오씨피","msql":"엠에스큐엘","koajs":"코아제이에스","lessjs":"레스제이에스","apachestruts":"아파치스트럿츠","blender":"블렌더","gis":"지아이에스","loopback":"루프백","apachedruid":"아파치드루이드"
                    ,"thymeleaft":"타임리프","sso":"싱글사인온","oracledatabase":"오라클데이터베이스","googlecloudsdk":"구글클라우드에스디케이","awselasticache":"아마존엘라스틱캐치","caffe":"카페","azureemotion":"애저이모션","lua":"루아","freebsd":"프리비에스디","googleapi":"구글에이피아이","jquerymobile":"제이쿼리모바일"
                    ,"googleappsscript":'구글앱스크립트','jwt':'제이슨웹토큰','googlecontainerregistry': '구글컨테이너레지스트리','rxjs':'알엑스제이에스','datavisualization': '데이터시각화','elasticstack': '엘라스틱스택',
                    'januswebrtcserver': '야누스웹알티서버','logstash': 'Logstash','kibana': '키바나','webflux': 'WebFlux','xplatform': '크로스 플랫폼',
'xml': '엑스엠엘',
'hypervisor': '하이퍼바이저',
'geoserver': '지오서버',
'awsmsk': '아마존엠에스케이',
'aws': '아마존',
'rdb': '관계형데이터베이스',
'adonisjs': '아도니스제이에스',
'react': '리액트',
'awsrdsforpostgresql': '아마존포스트그레에스큐엘',
'springframework': '스프링프레임워크',
'msa': '마이크로서비스아키텍처',
'springmvc': '스프링엠브이씨',
'golang': '고랭',
'googlecloudbigtable': '구글클라우드빅테이블',
'googleclouddataflow': '구글클라우드데이터플로우',
'spark': '스파크',
'k8s': '케이팔에스',
'cachet': '캐셋',
'kafka': '카프카',
'slack': '슬랙',
'grafana': '그라파나',
'prometheus': '프로메테우스',
'redux': '리덕스',
'recoil': '레코일',
'vite': '바이트',
'pytest': '파이테스트',
'airflow': '에어플로우',
'gitflow': '깃플로우',
'notion': '노션',
'elk': '이엘케이',
'sentry': '센트리',
'sequelize': '시큐얼라이즈',
'redmine': '레드마인',
'miplatform': '엠아이플랫폼',
'androidstudio': '안드로이드스튜디오',
'androidos': '안드로이드운영체제',
'eclipse': '이클립스',
'confluence': '컨플루언스',
'googledocs': '구글문서',
'erp': '이알피',
'db': '데이터베이스',
'bi': '비즈니스인텔리전스',
'json': '제이슨',
'djangorestframework': '장고레스트프레임워크',
'javaee': '자바이이',
'.netcore': '닷넷코어',
'network': '네트워크',
'dockerized': '도커화',
'flyway': '플라이웨이',
'nodal.js': '노달제이에스',
'hyperledgerfabric': '하이퍼레저페브릭',
'cassandra': '카산드라',
'swaggerui': '스웨거유아이',
'microsoftsqlserver': '마이크로소프트에스큐엘서버',
'oauth2': '오애쓰이',
'bitbucketpipelines': '빗버켓파이프라인',
'rails': '레일스',
'awsses': '아마존에스이에스',
'openapi': '오픈에이피아이',
'sw': '소프트웨어',
'fedora': '페도라',
'azurecontainerservice': '애저컨테이너서비스',
'awssqs': '아마존에스큐에스',
'ai/인공지능': '인공지능',
'vpn': '브이피엔',
'wsl': '더블유에스엘',
'visualstudiocode': '비주얼스튜디오코드',
'swaggerhub': '스웨거허브',
'react.jsboilerplate': '리액트제이에스보일러플레이트',
'notion.so': '노션',
'jupyter': '주피터',
'three.js': '쓰리제이에스',
'fpga': '에프피지에이',
'awsbatch': '아마존배치',
'apachemaven': '아파치메이븐',
'asp.netcore': '넷코어',
'awsshell': '아마존쉘',
'jqueryui': '제이쿼리유아이',
'smartcontract': '스마트계약',
'java8': '자바팔',
'azuresearch': '애저검색',
'awsecr': '아마존이씨알',
'nft': '엔에프티',
'intellijidea': '인텔리제이아이디어',
'awselasticsearchservice': '아마존엘라스틱검색서비스',
'argo': '아르고',
'cloudfirestore': '클라우드파이어스토어',
'visualc++': '비주얼씨플플',
'photon': '포톤',
'datatables': '데이터테이블',
'lucene': '루씬',
'hadoop': '하둡',
'awselastictranscoder': '아마존엘라스틱번역기',
'isms': '아이에스엠에스',
'gui': '지유아이',
'hive': '하이브',
'microsoftiis': '마이크로소프트아이아이에스',
'stream': '스트림',
'kafkastreams': '카프카스트림',
'awsdatapipeline': '아마존데이터파이프라인',
'vb.net': '비주얼베이직',
'awsaurora': '아마존오로라',
'druid': '드루이드',
'zookeeper': '주키퍼','abap': '에이벺',
'ibatis': '아이베티스','awsiotdevicemanagement': '사물인터넷디바이스관리','restli': '레스트엘이','hyperledgerindy': '하이퍼레저인디','apachecordova': '아파치코르도바',
'asana': '아사나',
'wiki.js': '위키',
'influxdb': '인플럭스디비',
  'figma': '피그마',
  'classicasp': '클래식에이에스피',
  'solr': '솔라',
  'azuredatafactory': '애저데이터팩토리',
  'phpstorm': '피에이치피스톰',
  'rpa': '알피에이',
  'reactquery': '리액트쿼리',
  'emotion': '이모션',
  'zustand': '저스트앤드',
  'jsonapi': '제이슨에이피아이',
  'phpci': '피에이치피씨아이',
  'postgis': '포스트지아이에스',
  'graphql.js': '그래프큐엘제이에스',
  'ips': '아이피에스',
  'visualstudio': '비주얼스튜디오',
  'defi': '디파이',
  'strapi': '스트래피아이',
  'macos': '맥오에스',
  'fabric': '패브릭',
  'mes': '메스',
  'azuredevops': '애저데브옵스',
  'apollo': '아폴로',
  'vercel': '버셀',
  'scss': '에스씨에스에스',
  'angular2': '앵귤러이',
  'microsoftteams': '마이크로소프트팀',
  'swr': '에스더블유알',
  'fastlane': '패스트라인',
  'protobuf': '프로토콜버퍼',
  'ionic': '아이오닉',
  'reactrouter': '리액트라우터',
  'createreactnativeapp': '크리에이트리액트네이티브앱',
  'chart.js': '차트제이에스',
  'semanticui': '시멘티씨유아이',
  'semanticuireact': '시맨티씨유아이리액트',
  'reactcanvas': '리액트캔버스',
  'awselasticloadbalancing': '아마존엘라스틱로드밸런싱',
  'antdesign': '앤트디자인',
  'prettier': '프리티어',
  'capacitor': '카파시터',
  'less': '레스',
  'entityframework': '엔티티프레임워크',
  'sketch': '스케치',
  'adobexd': '어도브엑스디',
  'adobeillustrator': '어도브일러스트레이터',
  'superset': '슈퍼셋',
  'webpacker': '웹패커',
  'ember.js': '엠버제이에스',
  'cloudflare': '클라우드플레어',
  'adobephotoshop': '어도브포토샵',
  'reactredux': '리액트리덕스',
  'babylonjs': '바빌론제이에스',
  'androidsdk': '안드로이드에스디케이',
  'vmware': '브이엠웨어',
  'platform.sh': '플랫폼에스에이치',
  'vuenative': '뷰네이티브',
  'playwright': '플레이라이트',
  'gitlabci': '깃랩씨아이',
  'githubenterprise': '깃랩엔터프라이즈',
  'vuetify': '뷰티파이',
  'mantis': '맨티스',
  'gerritcodereview': '제릿코드리뷰',
  'yii': '와이이',
  'redhatopenshift': '레드햇오픈시프트',
  'webflow': '웹플로우',
  'apachefreemarker': '아파치프리메이커',
  'apachetrafficserver': '아파치트래픽서버',
  'wordpress': '워드프레스',
  'orcad': '오카드',
  'h2database': '에이치투데이터베이스',
  'meteor': '메테오',
  'oraclepl/sql': '오라클피엘에스큐엘',
  'embedded': '임베디드',
  'embeddedlinux': '임베디드 리눅스',
  'reactivex': '리액티브엑스',
  'xcode': '엑스코드',
  'firebasehosting': '파이어베이스호스팅',
  'okhttp': '오케이에이치티피',
  'googledrive': '구글드라이브',
  'reactnavigation': '리액트네비게이션',
  'postman': '포스트맨',
  'googlecloudmessaging': '구글클라우드메세지',
  'crashlytics': '크래시틱스',
  'azureapplicationinsights': '애저애플리케이션인사이트',
  'swiftui': '스위프트유아이',
  'githubpages': '깃허브페이지',
  'materialdesign': '매테리얼디자인',
  'cocoatouch': '코코아터치',
  'nimble': '님블',
  'qa': '큐에이',
  'hasura': '하슈라',
  'facebookads': '페이스북에이디에스',
  'googleads': '구글에이디에스',
  'cocos2d-x': '코코스투디엑스',
  'webstorm': '웹스톰',
  '3drendering': '삼디렌더링',
  'hbase': '에이치베이스',
  'matplotlib': '맷플라립',
  'eda': '이디에이',
  'pycharm': '파이참',
  'olap': '오랩',
  'vba': '브이비에이',
  'powerbi': '파워비아이',
  'tibero': '티블로',
  'microsoftexcel': '마이크로소프트엑셀',
  'graphenedb': '그래픈데이터베이스',
  'microsoftoffice365': '마이크로소프트삼육오',
  'googletagmanager': '구글택매니저',
  'redash': '레드에쉬',
  'sqlalchemy': '에스큐엘알케미',
  'dw': '디더블유',
  'dm': '디엠',
  'datastudio': '데이터스튜디오',
  'etl': '이티엘',
  'dynamo': '다이나모',
  'scrapy': '스크래파이',
  'databricks': '데이터브릭스',
  'snowflake': '스노우플레이크',
  'awscloudhsm': '아마존클라우드에이치에스엠',
  'cisa': '씨사',
  'cissp': '씨아이에스에스피',
  'azurevirtualmachines': '애저가상머신',
  'awsmachinelearning': '아마존기계학습',
  'transformers': '트랜스포머스',
  'tensorflow.js': '텐서플로제이에스',
  'caffe2': '카페이',
  'awsappmesh': '아마존앱메쉬',
  '3dvolumerendering': '쓰리디볼륨렌더링',
  'kubeflow': '큐브플로우',
  'hw': '하드웨어',
  'fw': '펌웨어',
  'dialogflow': '다이어로그플로우',
  'mcu': '엠씨유',
  'pads': '패드스',
  'cloudcms': '클라우드씨엠에스',
  'zabbix': '자빅스',
  'vagrant': '바그란트',
  'arduino': '아두이노',
  'microsoftpowerapps': '마이크로소프트파워앱'
  ,'lowcodeengine': '로우코드엔진',
'microsoftpowerautomate': '마이크로소프트파워오토메이트',
'azureappservice': '애저앱서비스',
'ccna': '씨씨엔에이',
'plc': '피엘씨',
'firewall': '방화벽',
'xenserver': '젠서버',
'l2': '엘투',
'chef': '셰프',
'loki': '로키',
'kong': '콩',
'cisco': '시스코',
'l3': '엘쓰리',
'l4': '엘포',
'router': '라우터',
'switch': '스위치',
'gnubash': '지엔유배시',
'yocto': '요크토',
'l7': '엘칠',
'powershell': '파워셸',
'infra': '인프라',
'vault': '보안금고',
'helm': '헬름',
'bamboo': '밤부',
'utm': '유티엠',
'ibmcontainers': '아이비엠컨테이너',
'dynatrace': '다이나트레이스',
'dockercompose': '도커컴포즈',
'jboss': '제이보스',
'vmwarevsphere': '브이엠웨어스피어',
'azuredevopsserver': '애저데브옵스서버',
'azuresynapse': '애저시냅스',
'ldap': '엘디에이피',
'rf': '랜덤포레스트',
'awswaf': '아마존더블유에이에프',
'azurestorage': '애저스토리지',
'azurecosmosdb': '애저코스모스데이터베이스',
'detacloud': '데타클라우드',
'ossec': '오섹',
'phantomjs': '팬텀제이에스',
'ccnp': '씨씨엔피',
'cppg': '씨피피지',
'hack': '해킹',
'azuresecuritycenter': '애저보안센터',
'macosx': '맥오에스엑스',
'testng': '테스트엔지',
'testrail': '테스트레일',
'perforce': '퍼포스',
'fiddler': '피들러',
'jandi': '잔디',
'confluent': '콘플루언트',
'analog': '아날로그',
'asic': '에이식',
'axure': '액스쳐',
'dockercloud': '도커클라우드',
'paypal': '페이팔',
'pcb': '피씨비',
'eos': '이오스',
'msmq': '엠에스엠큐',
'salesforce': '세일즈포스',
'adobeexperiencemanager': '어도비익스피리언스매니저',
'circuitdesign': '회로설계',
'vhdl': '브이에이치디엘',
'ethernet': '이더넷',
'julia': '줄리아',
'autocad': '오토캐드',
'haskell': '해스켈',
'f#': '에프샵',
'perl': '펄',
'smps': '에스엠피에스',
'communication': '통신',
'azureiothub': '애저아이오티허브',
'pciproxy': '피씨아이프록시',
'apitracker': '에이피아이트래커',
'canjs': '캔제이에스',
'2drendering': '투디렌더링',
'goland': '고랜드',
'virtualbox': '가상박스',
'dicom': '디콤',
'vim': '빔',
'backendless': '백엔드레스',
'webcomponents': '웹컴포넌트',
'blazor': '블레이저',
'powerbuilder': '파워빌더',
'sourcetree': '소스트리',
'resharper': '리샤퍼',
'teamcity': '팀시티',
'testcomplete': '테스트컴플리트',
'vscode.dev': '브이에스코드데브',
'woocommerce': '우커머스',
'material': '마테리얼',
'microsoftoutlook': '마이크로소프트아웃룩',
'microsoftsharepoint': '마이크로소프트쉐어포인트',
'websphere': '웹스피어',
'redhatenterpriselinux': '레드햇엔터프라이즈리눅스',
'rhcsa': '알에이치씨에스에이',
'trello': '트렐로',
'blueprint': '블루프린트'}
        
    def add_term(self, term, definition):
        self.glossary[term.lower()] = definition
    def get_definition(self, term):
        return self.glossary.get(term.lower())
    def print_glossary(self):
        for term, definition in self.glossary.items():
            print(f"{term}: {definition}")
            
class ITGlossaryUpdater:
    def __init__(self, glossary):
        self.glossary = glossary
        self.url_list = ['https://wikidocs.net/67110', "https://wikidocs.net/67069", "https://wikidocs.net/67067", "https://wikidocs.net/67188", "https://wikidocs.net/67190"] # 1번째: 일반, 2번째: 프로그래밍, 데이터베이스, 3번째: 네트워크,클라우드,보안,블록체인, 4번째: 데이터과학, 인공지능, 통계, 5번째: 사물인터넷 등등 

    def update_glossary(self):
        for url in self.url_list:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
    
            for row in soup.select('table tbody tr'):
                term = row.select_one('td:nth-of-type(1)').text.strip()
                definition = row.select_one('td:nth-of-type(2)').text.strip()
                self.glossary.add_term(term, definition)