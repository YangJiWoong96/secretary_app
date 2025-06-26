import 'dart:async';
import 'dart:convert';
import 'dart:isolate';
import 'dart:developer'; // 로그를 위해 추가
import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:geolocator/geolocator.dart';
import 'package:device_calendar/device_calendar.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;

// 백그라운드 Task 진입점 (항상 top‐level)
@pragma('vm:entry-point')
void startCallback() {
  FlutterForegroundTask.setTaskHandler(MyTaskHandler());
}

// Future<void> main() async {
//   WidgetsFlutterBinding.ensureInitialized();

//   // 통신 포트 초기화 (TaskHandler ↔ UI)
//   FlutterForegroundTask.initCommunicationPort();

//   // Foreground Task 초기화
//   _initForegroundTask();

//   runApp(const MyApp());
// }

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await dotenv.load();  // 반드시 먼저 호출

  FlutterForegroundTask.initCommunicationPort();

  _initForegroundTask();

  runApp(const MyApp());
}

void _initForegroundTask() {
  FlutterForegroundTask.init(
    androidNotificationOptions: AndroidNotificationOptions(
      channelId: 'location_foreground',
      channelName: '위치 추적 서비스',
      channelDescription: '앱이 종료되어도 위치 추적을 지속합니다.',
      channelImportance: NotificationChannelImportance.LOW,
      priority: NotificationPriority.LOW,
      onlyAlertOnce: true,
    ),
    iosNotificationOptions: const IOSNotificationOptions(),
    foregroundTaskOptions: ForegroundTaskOptions(
      eventAction: ForegroundTaskEventAction.repeat(300000), // 5분 간격
      autoRunOnBoot: true,
      allowWakeLock: true,
      allowWifiLock: false,
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Timely Agent',
      theme: ThemeData(useMaterial3: true),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});
  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  Position? _currentPosition;
  StreamSubscription<Position>? _posStreamSub;
  StreamSubscription<dynamic>? _receivePortSub;
  List<Event> _events = [];

  @override
  void initState() {
    super.initState();
    _initService();
  }

  @override
  void dispose() {
    _posStreamSub?.cancel();
    _receivePortSub?.cancel();
    super.dispose();
  }

  Future<void> _initService() async {
    // 1) 위치 서비스 활성화
    if (!await Geolocator.isLocationServiceEnabled()) {
      await Geolocator.openLocationSettings();
      log('위치 서비스 비활성화로 설정 화면 이동');
      return;
    }

    // 2) 위치 권한 요청 (Geolocator + 항상 허용)
    var p = await Geolocator.checkPermission();
    if (p == LocationPermission.denied) {
      p = await Geolocator.requestPermission();
      log('위치 권한 요청 결과: $p');
    }
    if (p == LocationPermission.deniedForever) {
      log('위치 권한 영구 거부됨');
      return;
    }
    if (await Permission.locationAlways.request().isDenied) {
      log('항상 허용 위치 권한 필요');
      return;
    }

    // 3) UI 위치 스트림 구독
    _posStreamSub = Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.high, distanceFilter: 10),
    ).listen((pos) {
      setState(() => _currentPosition = pos);
      log('UI 위치 업데이트: ${pos.latitude}, ${pos.longitude}');
    });

    // 4) 캘린더 권한 요청
    final calPerm = await DeviceCalendarPlugin().requestPermissions();
    log('캘린더 권한 요청 결과: $calPerm');

    // 5) Foreground 서비스 시작
    final started = await FlutterForegroundTask.startService(
      serviceId: 1,
      notificationTitle: 'Timely Agent',
      notificationText: '백그라운드에서 실행 중입니다.',
      callback: startCallback,
    );
    log('ForegroundService 시작: $started');

    // 6) 백그라운드 통신 구독
    final rawPort = FlutterForegroundTask.receivePort;
    if (rawPort != null && _receivePortSub == null) {
      _receivePortSub = rawPort.asBroadcastStream().listen((data) {
        if (data is Map<String, dynamic> && mounted) {
          setState(() {
            _events = (data['events'] as List)
                .map((e) => Event.fromJson(e))
                .toList();
          });
          log('백그라운드 이벤트 수신: ${_events.length}개');
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Timely Agent')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(_currentPosition != null
                ? '📍 위치: ${_currentPosition!.latitude.toStringAsFixed(4)}, ${_currentPosition!.longitude.toStringAsFixed(4)}'
                : '위치 정보 수집 대기 중...'),
            const SizedBox(height: 20),
            const Text('📅 다가오는 일정:'),
            const Divider(),
            Expanded(
              child: _events.isEmpty
                  ? const Center(child: Text('없음'))
                  : ListView.builder(
                      itemCount: _events.length,
                      itemBuilder: (_, i) {
                        final e = _events[i];
                        return Card(
                          margin: const EdgeInsets.symmetric(vertical: 4),
                          child: ListTile(
                            leading: const Icon(Icons.event_note),
                            title: Text(e.title ?? ''),
                            subtitle: Text(e.start?.toLocal().toString() ?? ''),
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

class MyTaskHandler extends TaskHandler {
  StreamSubscription<Position>? _posSub;
  Position? _lastPos;

  @override
  Future<void> onStart(DateTime timestamp, TaskStarter starter) async {
    log('TaskHandler onStart: $timestamp');
    _posSub = Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.medium, distanceFilter: 50),
    ).listen(_processData);
  }

  Future<void> _processData(Position p) async {
    final distance = Geolocator.distanceBetween(
      _lastPos?.latitude ?? 0,
      _lastPos?.longitude ?? 0,
      p.latitude,
      p.longitude,
    );
    log('MyTaskHandler: _processData 실행. _lastPos is null: ${_lastPos == null}. Distance: $distance');

    // 앱이 죽는 것을 막기 위해 에러 핸들링 추가
    List<Event> evs = [];
    try {
      evs = await _fetchCalendarEvents();
      log('다가오는 일정 불러옴: ${evs.length}개');
    } catch (e) {
      log('일정 불러오기 중 에러 발생: $e');
    }


    FlutterForegroundTask.sendDataToMain({
      'position': p.toJson(),
      'events': evs.map((e) => e.toJson()).toList(),
    });

    if (_lastPos == null || distance > 30) {
      log('MyTaskHandler: 전송 조건 만족! _sendDataToBackend 호출 시도.');
      _lastPos = p;
      await _sendDataToBackend(p, evs);
    } else {
      log('MyTaskHandler: 이동거리 미충족 (30m 이하)');
    }
  }

  Future<void> _sendDataToBackend(Position p, List<Event> evs) async {
    final now = DateTime.now();
    final payload = {
      "userId": "test_user_01",
      "recordTime": now.toIso8601String(),
      "contextData": {
        "location": {
          "latitude": p.latitude,
          "longitude": p.longitude,
          "timestamp": (p.timestamp ?? now).toIso8601String(),
        },
        "upcomingEvents": evs
            .map((e) => {
                  "title": e.title ?? "",
                  "startTime": e.start?.toIso8601String() ?? "",
                  "endTime": e.end?.toIso8601String() ?? "",
                  "location": e.location ?? "",
                })
            .toList(),
      }
    };
    log('▶ 서버 전송 페이로드: ${jsonEncode(payload)}');

    try {
      final res = await http.post(
        Uri.parse('${dotenv.env['SERVER_URL']}/api/v1/context'),
        headers: {'Content-Type': 'application/json; charset=UTF-8'},
        body: jsonEncode(payload),
      );
      log('✅ 요청 결과: ${res.statusCode}, 응답: ${res.body}');
    } catch (e) {
      log('❌ 네트워크 또는 기타 에러: $e');
    }
  }

// Uri.parse('http://52.63.186.70:8000/api/v1/context')

  Future<List<Event>> _fetchCalendarEvents() async {
    final plugin = DeviceCalendarPlugin();
    final cals = await plugin.retrieveCalendars();
    if (cals.isSuccess && cals.data != null) {
      final list = <Event>[];
      for (var cal in cals.data!) {
        // ✅✅✅✅✅ 여기가 수정된 핵심 부분 ✅✅✅✅✅
        if (cal.id != null) {
          final evs = await plugin.retrieveEvents(
            cal.id!, // 이제 안전합니다.
            RetrieveEventsParams(
              startDate: DateTime.now(),
              endDate: DateTime.now().add(const Duration(days: 1)),
            ),
          );
          if (evs.isSuccess && evs.data != null) {
            list.addAll(evs.data!);
          }
        }
      }
      return list;
    }
    log('캘린더 일정 조회 실패 또는 데이터 없음');
    return [];
  }

  @override
  void onRepeatEvent(DateTime timestamp) {
    log('onRepeatEvent: $timestamp');
  }

  @override
  Future<void> onDestroy(DateTime timestamp, bool isTimeout) async {
    log('TaskHandler onDestroy: timeout=$isTimeout');
    await _posSub?.cancel();
  }
}