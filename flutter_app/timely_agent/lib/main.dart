import 'dart:async';
import 'dart:convert';
import 'dart:isolate';
import 'package:flutter/material.dart';
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

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // 통신 포트 초기화 (TaskHandler ↔ UI)
  FlutterForegroundTask.initCommunicationPort();

  // Foreground Task 초기화
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
      return;
    }

    // 2) 위치 권한 요청 (Geolocator + 항상 허용)
    var p = await Geolocator.checkPermission();
    if (p == LocationPermission.denied) p = await Geolocator.requestPermission();
    if (p == LocationPermission.deniedForever) {
      debugPrint("권한 거부됨");
      return;
    }
    // permission_handler로 항상 허용 권한 요청
    if (await Permission.locationAlways.request().isDenied) {
      debugPrint("항상 허용 위치 권한 필요");
      return;
    }

    // 3) UI 쪽 위치 스트림 구독
    _posStreamSub = Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.high, distanceFilter: 10),
    ).listen((pos) => setState(() => _currentPosition = pos));

    // 4) 캘린더 권한 요청
    await DeviceCalendarPlugin().requestPermissions();

    // 5) Foreground 서비스 시작 (앱 종료해도 유지)
    await FlutterForegroundTask.startService(
      serviceId: 1,
      notificationTitle: 'Timely Agent',
      notificationText: '백그라운드에서 실행 중입니다.',
      callback: startCallback,
    );

    // 6) 백그라운드 → 메인 통신 구독 (broadcast)
    final rawPort = FlutterForegroundTask.receivePort;
    if (rawPort != null && _receivePortSub == null) {
      _receivePortSub = rawPort.asBroadcastStream().listen((data) {
        if (data is Map<String, dynamic> && mounted) {
          setState(() {
            _events = (data['events'] as List)
                .map((e) => Event.fromJson(e))
                .toList();
          });
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
                ? '📍 위치: ${_currentPosition!.latitude.toStringAsFixed(4)}, '
                  '${_currentPosition!.longitude.toStringAsFixed(4)}'
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
                            subtitle:
                                Text(e.start?.toLocal().toString() ?? ''),
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
    _posSub = Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.medium, distanceFilter: 50),
    ).listen(_processData);
  }

  Future<void> _processData(Position p) async {
    final evs = await _fetchCalendarEvents();
    FlutterForegroundTask.sendDataToMain({
      'position': p.toJson(),
      'events': evs.map((e) => e.toJson()).toList(),
    });
    if (_lastPos == null ||
        Geolocator.distanceBetween(
                _lastPos!.latitude, _lastPos!.longitude, p.latitude, p.longitude) >
            30) {
      _lastPos = p;
      await _sendDataToBackend(p, evs);
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
    try {
      final res = await http.post(
        Uri.parse('http://10.0.2.2:8000/api/v1/context'),
        headers: {'Content-Type': 'application/json; charset=UTF-8'},
        body: jsonEncode(payload),
      );
      debugPrint(res.statusCode == 200 ? "✅ 성공" : "❌ ${res.statusCode}");
    } catch (e) {
      debugPrint("❌ 네트워크 에러: $e");
    }
  }

  Future<List<Event>> _fetchCalendarEvents() async {
    final plugin = DeviceCalendarPlugin();
    final cals = await plugin.retrieveCalendars();
    if (cals.isSuccess && cals.data != null) {
      final list = <Event>[];
      for (var cal in cals.data!) {
        final evs = await plugin.retrieveEvents(
          cal.id!,
          RetrieveEventsParams(
            startDate: DateTime.now(),
            endDate: DateTime.now().add(const Duration(days: 1)),
          ),
        );
        if (evs.isSuccess && evs.data != null) list.addAll(evs.data!);
      }
      return list;
    }
    return [];
  }

  @override
  void onRepeatEvent(DateTime timestamp) {}

  @override
  Future<void> onDestroy(DateTime timestamp, bool isTimeout) async {
    await _posSub?.cancel();
  }
}