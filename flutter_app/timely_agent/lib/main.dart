import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'dart:developer';
import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:geolocator/geolocator.dart';
import 'package:device_calendar/device_calendar.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:flutter_foreground_task/flutter_foreground_task.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sleek_circular_slider/sleek_circular_slider.dart';
import 'package:geolocator_android/geolocator_android.dart';
import 'package:geolocator_apple/geolocator_apple.dart';
import 'package:timezone/data/latest.dart' as tzdata;
import 'package:timezone/timezone.dart' as tz;

Future<void> _requestIgnoreBatteryOptimizations() async {
  final isIgnoring = await FlutterForegroundTask.isIgnoringBatteryOptimizations;
  if (!isIgnoring) {
    // 시스템 설정화면으로 이동
    await FlutterForegroundTask.openIgnoreBatteryOptimizationSettings();
  }
}

// 백그라운드 Task 진입점 (항상 top-level)
@pragma('vm:entry-point')
void startCallback() {
  log('[BACKGROUND LOG] startCallback initiated.'); // LOG
  if (Platform.isAndroid) {
    GeolocatorAndroid.registerWith();
  } else if (Platform.isIOS) {
    GeolocatorApple.registerWith();
  }
  FlutterForegroundTask.setTaskHandler(MyTaskHandler());
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await dotenv.load();
  await _requestIgnoreBatteryOptimizations();
  tzdata.initializeTimeZones();
  _initForegroundTask();
  log('[MAIN LOG] main() function finished, running app.'); // LOG
  runApp(const MyApp());
}

void _initForegroundTask() {
  log('[MAIN LOG] _initForegroundTask() called.'); // LOG
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
      eventAction: ForegroundTaskEventAction.repeat(1800000),
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

class _MyHomePageState extends State<MyHomePage> with WidgetsBindingObserver {
  // 실시간 UI용 위치 스트림 구독
  StreamSubscription<Position>? _posSubUI;
  Position? _currentPosition;
  List<Event> _events = [];
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    print('[UI LOG] initState() called.'); // LOG
    WidgetsBinding.instance.addObserver(this);
    _initializeApp();
    _timer = Timer.periodic(const Duration(seconds: 15), (timer) {
      print('[UI LOG] 15-second timer fired, calling _updateDataFromPrefs.'); // LOG
      _updateDataFromPrefs();
    });
  }

  void _showRestartDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => AlertDialog(
        title: const Text('권한 적용을 위해 재시작 필요'),
        content: const Text('모든 권한이 정상적으로 적용되었습니다. 앱을 완전히 종료한 후 다시 실행해 주세요.'),
        actions: [
          TextButton(
            onPressed: () => exit(0),
            child: const Text('앱 종료'),
          ),
        ],
      ),
    );
  }

  // _MyHomePageState 클래스 최상단에 잠금 변수 추가
  bool _isInitializing = false;

  Future<void> _initializeApp() async {
    // --- ⭐️ 중복 실행 방지 잠금(Lock) ⭐️ ---
    if (_isInitializing) {
      print('[UI LOG] Initialization is already in progress. Skipping call.');
      return;
    }
    _isInitializing = true;
    print('[UI LOG] _initializeApp() started.');

    try {
      final prefs = await SharedPreferences.getInstance();
      final isPreferenceSet = prefs.getBool('pref_completed') ?? false;
      print('[UI LOG] Preference Set: $isPreferenceSet');

      if (!isPreferenceSet) {
        if (mounted) {
          print('[UI LOG] Navigating to PreferenceScreen.');
          await Navigator.of(context).push(
            MaterialPageRoute(builder: (_) => const PreferenceScreen()),
          );
        }
      }

      final isLocationServiceEnabled = await Geolocator.isLocationServiceEnabled();
      print('[UI LOG] Location Service Enabled: $isLocationServiceEnabled');
      if (!isLocationServiceEnabled) {
        await Geolocator.openLocationSettings();
        _isInitializing = false; // 함수 종료 전 잠금 해제
        return;
      }

      var locationPermission = await Geolocator.checkPermission();
      print('[UI LOG] Initial Location Permission Status: $locationPermission');
      if (locationPermission == LocationPermission.denied) {
        locationPermission = await Geolocator.requestPermission();
        print('[UI LOG] Requested Location Permission, New Status: $locationPermission');
      }
      if (locationPermission == LocationPermission.denied ||
          locationPermission == LocationPermission.deniedForever) {
        if (mounted) _showPermissionDialog('위치');
        _isInitializing = false; // 함수 종료 전 잠금 해제
        return;
      }

      final locationAlwaysStatus = await Permission.locationAlways.request();
      print('[UI LOG] Location Always Permission Status: $locationAlwaysStatus');
      if (!locationAlwaysStatus.isGranted) {
        if (mounted) _showPermissionDialog('백그라운드 위치');
        _isInitializing = false; // 함수 종료 전 잠금 해제
        return;
      }

      final notificationStatus = await Permission.notification.request();
      print('[UI LOG] Notification Permission Status: $notificationStatus');
      if (!notificationStatus.isGranted) {
        if (mounted) _showPermissionDialog('알림');
        _isInitializing = false; // 함수 종료 전 잠금 해제
        return;
      }

      final bool hasGrantedCalendarBefore =
          prefs.getBool('calendar_permission_granted_once') ?? false;
      print('[UI LOG] Has granted calendar permission before: $hasGrantedCalendarBefore');

      final calendarPermissionResult =
          await DeviceCalendarPlugin().requestPermissions();
      final bool isCalendarGrantedNow =
          calendarPermissionResult.isSuccess && (calendarPermissionResult.data ?? false);
      print('[UI LOG] Calendar Permission Request Result - isSuccess: ${calendarPermissionResult.isSuccess}, data: ${calendarPermissionResult.data}');

      if (isCalendarGrantedNow && !hasGrantedCalendarBefore) {
        print('[UI LOG] Calendar permission JUST granted. Requiring restart.');
        await prefs.setBool('calendar_permission_granted_once', true);
        if (mounted) _showRestartDialog();
        _isInitializing = false; // 함수 종료 전 잠금 해제
        return;
      } else if (!isCalendarGrantedNow) {
        print('[UI LOG] Calendar permission is NOT granted.');
        if (mounted) _showPermissionDialog('캘린더');
        _isInitializing = false; // 함수 종료 전 잠금 해제
        return;
      }

      print('[UI LOG] All permissions seem to be in order. Fetching initial data.');
      await _fetchInitialData();

      final isRunning = await FlutterForegroundTask.isRunningService;
      print('[UI LOG] Is Foreground Service Running? $isRunning');
      if (!isRunning) {
        await _startForegroundService();
      }
      print('[UI LOG] _initializeApp() finished.');
    } finally {
      // --- ⭐️ 모든 로직이 끝나면 항상 잠금 해제 ⭐️ ---
      _isInitializing = false;
    }
  }

  Future<void> _fetchInitialData() async {
    print('[UI LOG] _fetchInitialData() started.'); // LOG
    try {
      final position = await Geolocator.getCurrentPosition();
      print('[UI LOG] Fetched current position: $position'); // LOG
      if (mounted) setState(() => _currentPosition = position);
    } catch (e) {
      print("[UI LOG] ERROR fetching current position: $e"); // LOG
    }

    await _updateDataFromPrefs();
    
    print('[UI LOG] Starting location stream listener.'); // LOG
    _startListeningToLocationUpdates();
  }

  void _startListeningToLocationUpdates() {
    _posSubUI?.cancel();
    _posSubUI = Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 50,
      ),
    ).listen(
      (pos) {
        print('[UI LOG] Location stream update received: $pos'); // LOG
        if (mounted) setState(() => _currentPosition = pos);
      },
      onError: (e) {
        print("[UI LOG] ERROR in location stream: $e"); // LOG
      },
    );
  }

  Future<void> _startForegroundService() async {
    print('[UI LOG] Attempting to start foreground service.'); // LOG
    await FlutterForegroundTask.startService(
      serviceId: 1,
      notificationTitle: 'Timely Agent',
      notificationText: '백그라운드 실행 중',
      callback: startCallback,
    );
    print('[UI LOG] Foreground service start command issued.'); // LOG
  }
  
  @override
  void dispose() {
    print('[UI LOG] dispose() called.'); // LOG
    _timer?.cancel();
    _posSubUI?.cancel();
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    print('[UI LOG] didChangeAppLifecycleState changed to: $state'); // LOG
    if (state == AppLifecycleState.resumed) {
      _initializeApp();
    }
  }

  Future<void> _updateDataFromPrefs() async {
    print('[UI LOG] _updateDataFromPrefs() started.'); // LOG
    final prefs = await SharedPreferences.getInstance();
    if (!mounted) {
      print('[UI LOG] _updateDataFromPrefs: not mounted, exiting.'); // LOG
      return;
    }

    // final posString = prefs.getString('latest_position');
    // if (posString != null) {
    //   _currentPosition = Position.fromMap(jsonDecode(posString));
    // }

    final eventString = prefs.getString('latest_events');
    if (eventString != null) {
      print('[UI LOG] Found events in SharedPreferences. Parsing...'); // LOG
      final raw = jsonDecode(eventString) as List<dynamic>;
      _events = raw.map((item) {
        final startStr = item['start']?.toString() ?? '';
        final endStr = item['end']?.toString() ?? '';
        final startDt = DateTime.tryParse(startStr);
        final endDt = DateTime.tryParse(endStr);
        final tzStart =
            startDt == null ? null : tz.TZDateTime.from(startDt, tz.local);
        final tzEnd =
            endDt == null ? null : tz.TZDateTime.from(endDt, tz.local);
        return Event('')
          ..title = item['title']?.toString()
          ..start = tzStart
          ..end = tzEnd
          ..location = item['location']?.toString();
      }).toList();
      print('[UI LOG] Parsed ${raw.length} events from SharedPreferences.'); // LOG
    } else {
      print('---- 🔍 [UI LOG] No events in SharedPreferences. Fetching directly. ----'); // LOG
      try {
        final plugin = DeviceCalendarPlugin();
        final permissionsGranted = await plugin.hasPermissions();
        print('[UI LOG] Calendar permission status: ${permissionsGranted.data}'); // LOG
        if (permissionsGranted.isSuccess && permissionsGranted.data == true) {
          final cals = await plugin.retrieveCalendars();
          if (cals.isSuccess && cals.data != null) {
            print('🗓️ [UI LOG] Found ${cals.data!.length} calendars.'); // LOG
            List<Event> list = [];
            for (var cal in cals.data!) {
              print('   -> [UI LOG] Processing calendar: ${cal.name} (ID: ${cal.id})'); // LOG
              if (cal.id != null) {
                final evs = await plugin.retrieveEvents(
                  cal.id!,
                  RetrieveEventsParams(
                    startDate: DateTime.now(),
                    endDate: DateTime.now().add(const Duration(days: 7)),
                  ),
                );
                if (evs.isSuccess && evs.data != null) {
                   print('       => [UI LOG] Found ${evs.data!.length} events in this calendar.'); // LOG
                  list.addAll(evs.data!);
                } else {
                   print('       => [UI LOG] Failed to get events from this calendar: ${evs.errors}'); // LOG
                }
              }
            }
            _events = list;
            print('✨ [UI LOG] Total events fetched: ${_events.length}'); // LOG
          } else {
             print('🚨 [UI LOG] Calendar retrieval failed or data is null: ${cals.errors}'); // LOG
          }
        } else {
          print('🚨 [UI LOG] Calendar permission check returned false.'); // LOG
        }
      } catch (e) {
        print("catastrophical [UI LOG] CRITICAL ERROR during direct calendar fetch: $e"); // LOG
      }
      print('---- 🔍 [UI LOG] Direct calendar fetch finished. ----'); // LOG
    }

    if (mounted) {
      print('[UI LOG] Calling setState() to update UI.'); // LOG
      setState(() {});
    }
  }

  void _showPermissionDialog(String permissionName) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: Text('$permissionName 권한 필요'),
        content: Text('앱의 모든 기능을 사용하려면 $permissionName 권한을 허용해야 합니다. 설정으로 이동하시겠습니까?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('취소')),
          TextButton(
            onPressed: () {
              openAppSettings();
              Navigator.pop(context);
            },
            child: const Text('설정 열기'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AYN'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => Navigator.of(
              context,
            ).push(MaterialPageRoute(builder: (_) => const PreferenceScreen())),
            tooltip: 'MyPage (선호도 설정)',
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              _currentPosition != null
                  ? '📍 ${_currentPosition!.latitude.toStringAsFixed(4)}, ${_currentPosition!.longitude.toStringAsFixed(4)}'
                  : '위치 대기 중',
            ),
            const SizedBox(height: 20),
            const Text('📅 다가오는 일정'),
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

// PreferenceScreen and ScheduleScreen classes are unchanged
class PreferenceScreen extends StatefulWidget {
  const PreferenceScreen({super.key});
  @override
  State<PreferenceScreen> createState() => _PreferenceScreenState();
}

class _PreferenceScreenState extends State<PreferenceScreen> {
  // 1. 새로운 컨트롤러 리스트 선언
  final _formKey = GlobalKey<FormState>();
  // 각 선호도 항목을 관리할 컨트롤러 쌍 리스트
  List<Map<String, TextEditingController>> _preferenceControllers = [];

  @override
  void initState() {
    super.initState();
    _loadPrefs();
  }

  Future<void> _loadPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    final prefString = prefs.getString('user_preferences');
    // 이전에 저장된 값이 없다면 기본 3개 항목으로 초기화
    if (prefString == null) {
      _preferenceControllers = [
        {'label': TextEditingController(text: '좋아하는 음식'), 'value': TextEditingController()},
        {'label': TextEditingController(text: '취미생활'), 'value': TextEditingController()},
        {'label': TextEditingController(text: '관심사'), 'value': TextEditingController()},
      ];
    } else {
      // 저장된 값이 있다면 JSON을 디코딩하여 컨트롤러 리스트 생성
      final Map<String, dynamic> prefMap = jsonDecode(prefString);
      _preferenceControllers = prefMap.entries.map((entry) {
        return {
          'label': TextEditingController(text: entry.key),
          'value': TextEditingController(text: entry.value.toString()),
        };
      }).toList();
    }
    setState(() {});
  }

  Future<void> _save(bool completed) async {
    final prefs = await SharedPreferences.getInstance();
      // 컨트롤러 리스트를 {'label': 'value'} 형태의 Map으로 변환
    final Map<String, String> preferencesMap = {
      for (var item in _preferenceControllers)
        if (item['label']!.text.isNotEmpty) // 라벨이 비어있지 않은 항목만 저장
          item['label']!.text: item['value']!.text
    };

    // Map을 JSON 문자열로 변환하여 저장
    await prefs.setString('user_preferences', jsonEncode(preferencesMap));
    await prefs.setBool('pref_completed', completed); 

    // 백엔드로 변환된 Map을 전달
    await _sendPreferencesToBackend(preferencesMap);
    if (mounted) Navigator.of(context).pop();
  }
  // Map을 받아 백엔드로 전송하는 _sendPreferencesToBackend 메소드
  Future<void> _sendPreferencesToBackend(Map<String, String> preferencesMap) async {
    final now = DateTime.now();
    final payload = {
      'userId': 'test_user_01',
      // 'preferences' 필드에 고정된 객체 대신 Map을 그대로 전달
      'preferences': preferencesMap,
      'updateTime': now.toIso8601String(),
    };
    try {
      final res = await http.post(
        Uri.parse('${dotenv.env['SERVER_URL']}/api/v1/preferences'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );
      print('✅ 선호도 업데이트 결과: ${res.statusCode}');
    } catch (e) {
      print('❌ 선호도 업데이트 오류: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('설정')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Expanded(
              child: Form(
                key: _formKey,
                child: Column(
                  children: [
                    const Text(
                      '앱 사용을 위해 기본 선호도와 추적 스케줄을 입력해주세요. 나중에 언제든 변경 가능',
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    Expanded(
                      child: ListView.builder(
                        itemCount: _preferenceControllers.length,
                        itemBuilder: (context, index) {
                          bool isDefaultItem = index < 3;
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 12.0),
                            child: Row(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                // --- 1. 라벨 입력 필드 ---
                                Expanded(
                                  flex: 2,
                                  child: TextFormField(
                                    controller: _preferenceControllers[index]['label'],
                                    readOnly: isDefaultItem,
                                    decoration: const InputDecoration(
                                      labelText: '항목',
                                      border: OutlineInputBorder(),
                                    ),
                                    validator: (value) {
                                      if (value == null || value.trim().isEmpty) {
                                        return '항목을 입력하세요';
                                      }
                                      return null;
                                    },
                                  ),
                                ),
                                const SizedBox(width: 8),
                                // --- 2. 내용 입력 필드 ---
                                Expanded(
                                  flex: 3,
                                  child: TextFormField(
                                    controller: _preferenceControllers[index]['value'],
                                    decoration: const InputDecoration(
                                      labelText: '내용',
                                      border: OutlineInputBorder(),
                                    ),
                                  ),
                                ),
                                // --- 3. 삭제 버튼 ---
                                if (!isDefaultItem)
                                  Padding(
                                    padding: const EdgeInsets.only(left: 4.0),
                                    child: IconButton(
                                      icon: const Icon(Icons.remove_circle_outline),
                                      onPressed: () {
                                        setState(() {
                                          _preferenceControllers.removeAt(index);
                                        });
                                      },
                                    ),
                                  )
                                else // 삭제 버튼이 없을 때 공간을 맞추기 위한 위젯
                                  const SizedBox(width: 52),
                              ],
                            ),
                          );
                        },
                      ),
                    ),
                    // --- 항목 추가 버튼 ---
                    if (_preferenceControllers.length < 10)
                      TextButton.icon(
                        icon: const Icon(Icons.add),
                        label: const Text('선호도 항목 추가'),
                        onPressed: () {
                          setState(() {
                            _preferenceControllers.add({
                              'label': TextEditingController(),
                              'value': TextEditingController(),
                            });
                          });
                        },
                      ),
                    const SizedBox(height: 10),
                    // --- 스케줄 설정 버튼 ---
                    ElevatedButton(
                      onPressed: () => Navigator.of(context).push(
                        MaterialPageRoute(
                          builder: (_) => const ScheduleScreen(),
                        ),
                      ),
                      child: const Text('추적 스케줄 설정'),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            // --- 저장/건너뛰기 버튼 ---
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: () {
                      if (_formKey.currentState!.validate()) {
                        _save(true);
                      }
                    },
                    child: const Text('저장'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton(
                    onPressed: () => _save(false),
                    child: const Text('건너뛰기'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class ScheduleScreen extends StatefulWidget {
  const ScheduleScreen({super.key});
  @override
  State<ScheduleScreen> createState() => _ScheduleScreenState();
}

class _ScheduleScreenState extends State<ScheduleScreen> {
  Map<int, List<TimeRange>> _schedule = {};

  @override
  void initState() {
    super.initState();
    _loadSchedule();
  }

  Future<void> _loadSchedule() async {
    final prefs = await SharedPreferences.getInstance();
    final str = prefs.getString('schedule');
    if (str != null) {
      final Map<String, dynamic> data = jsonDecode(str);
      setState(() {
        _schedule = data.map((day, ranges) {
          final wd = _dayKeyToWeekday(day);
          final list = (ranges as List).map((r) => _parseRange(r)).toList();
          return MapEntry(wd, list);
        });
      });
    } else {
      for (var i = 1; i <= 7; i++) _schedule[i] = [];
    }
  }

  Future<void> _saveSchedule() async {
    final prefs = await SharedPreferences.getInstance();
    final Map<String, List<String>> out = {};
    _schedule.forEach((wd, list) {
      out[_weekdayToDayKey(wd)] = list.map((r) => r.toString()).toList();
    });
    await prefs.setString('schedule', jsonEncode(out));
  }

  int _dayKeyToWeekday(String key) {
    const map = {
      'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7,
    };
    return map[key] ?? 1;
  }

  String _weekdayToDayKey(int wd) {
    const map = {
      1: 'mon', 2: 'tue', 3: 'wed', 4: 'thu', 5: 'fri', 6: 'sat', 7: 'sun',
    };
    return map[wd]!;
  }

  TimeRange _parseRange(String s) {
    final parts = s.split('-');
    final start = parts[0].trim().split(':');
    final end = parts[1].trim().split(':');
    return TimeRange(
      int.parse(start[0]) * 60 + int.parse(start[1]),
      int.parse(end[0]) * 60 + int.parse(end[1]),
    );
  }

  int _totalMinutes(int wd, {int? skipIndex, TimeRange? extra}) {
    var sum = 0;
    for (var i = 0; i < _schedule[wd]!.length; i++) {
      if (i == skipIndex) continue;
      sum += _schedule[wd]![i].duration;
    }
    if (extra != null) sum += extra.duration;
    return sum;
  }

  Future<void> _editRange(int wd, int index) async {
    final orig = _schedule[wd]![index];
    int startMin = orig.startMinutes;
    int endMin = orig.endMinutes;
    int step = 0;
    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (ctx, setSt) {
            return Padding(
              padding: EdgeInsets.only(
                bottom: MediaQuery.of(ctx).viewInsets.bottom + 16,
                top: 16, left: 16, right: 16,
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    step == 0 ? '시작 시간 선택' : '종료 시간 선택',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  SleekCircularSlider(
                    min: 0,
                    max: 24,
                    initialValue: (step == 0 ? startMin : endMin) / 60,
                    appearance: CircularSliderAppearance(
                      size: 250,
                      angleRange: 300,
                      startAngle: 120,
                      customWidths: CustomSliderWidths(
                        trackWidth: 8,
                        progressBarWidth: 12,
                        handlerSize: 12,
                      ),
                      infoProperties: InfoProperties(
                        mainLabelStyle: const TextStyle(fontSize: 24),
                        modifier: (val) =>
                            '${val.floor().toString().padLeft(2, '0')}:00',
                      ),
                    ),
                    onChangeEnd: (val) {
                      final m = (val.floor() * 60);
                      setSt(() {
                        if (step == 0)
                          startMin = m;
                        else
                          endMin = m;
                      });
                    },
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: () {
                      if (step == 0) {
                        step = 1;
                        setSt(() {});
                      } else {
                        if (endMin <= startMin) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('종료 시간이 시작 시간 이후여야 합니다.'),
                            ),
                          );
                          return;
                        }
                        if (_totalMinutes(
                              wd,
                              skipIndex: index,
                              extra: TimeRange(startMin, endMin),
                            ) > 960) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('총 누적 시간이 16시간을 초과합니다.'),
                            ),
                          );
                          return;
                        }
                        _schedule[wd]![index] = TimeRange(startMin, endMin);
                        _saveSchedule();
                        Navigator.of(ctx).pop();
                      }
                    },
                    child: Text(step == 0 ? '다음' : '확인'),
                  ),
                  const SizedBox(height: 8),
                ],
              ),
            );
          },
        );
      },
    );
    setState(() {});
  }

  Future<void> _addRange(int wd) async {
    if (_schedule[wd]!.length >= 3) return;
    final defaultRange = TimeRange(9 * 60, 17 * 60);
    if (_totalMinutes(wd, extra: defaultRange) > 960) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('16시간을 초과하여 추가할 수 없습니다.')));
      return;
    }
    setState(() => _schedule[wd]!.add(defaultRange));
    await _saveSchedule();
  }

  Future<void> _removeRange(int wd, int idx) async {
    setState(() => _schedule[wd]!.removeAt(idx));
    await _saveSchedule();
  }

  @override
  Widget build(BuildContext context) {
    final days = ['월', '화', '수', '목', '금', '토', '일'];
    return Scaffold(
      appBar: AppBar(title: const Text('추적 스케줄 설정')),
      body: ListView.builder(
        itemCount: 7,
        itemBuilder: (_, i) {
          final wd = i + 1;
          final ranges = _schedule[wd] ?? [];
          return ExpansionTile(
            title: Text('${days[i]}요일'),
            children: [
              for (var j = 0; j < ranges.length; j++)
                ListTile(
                  title: Text(ranges[j].toString()),
                  onTap: () => _editRange(wd, j),
                  trailing: IconButton(
                    icon: const Icon(Icons.delete),
                    onPressed: () => _removeRange(wd, j),
                  ),
                ),
              const Divider(),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: ElevatedButton.icon(
                  onPressed: ranges.length < 3 ? () => _addRange(wd) : null,
                  icon: const Icon(Icons.add),
                  label: const Text('시간대 추가'),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class TimeRange {
  final int startMinutes;
  final int endMinutes;
  TimeRange(this.startMinutes, this.endMinutes);
  int get duration => endMinutes - startMinutes;
  bool contains(int minutes) =>
      minutes >= startMinutes && minutes <= endMinutes;
  @override
  String toString() {
    final sh = (startMinutes ~/ 60).toString().padLeft(2, '0');
    final sm = (startMinutes % 60).toString().padLeft(2, '0');
    final eh = (endMinutes ~/ 60).toString().padLeft(2, '0');
    final em = (endMinutes % 60).toString().padLeft(2, '0');
    return '$sh:$sm-$eh:$em';
  }
}

class MyTaskHandler extends TaskHandler {
  StreamSubscription<Position>? _posSub;
  Position? _lastPos;
  int _currentThreshold = 300;
  DateTime _lastMajorMoveTime = DateTime.now(); // 45초, 20분 타임아웃 기준 시간
  DateTime? _last2kmMoveTime;                  // 2km 모드의 6분 타임아웃 기준 시간
  int _consecutive2kmCount = 0;
  Map<int, List<TimeRange>> _schedule = {};
  List<Event> _lastSentEvents = [];

    // ⭐️ SharedPreferences에 데이터를 저장하는 메서드
  Future<void> _saveDataToPrefs({Position? position, List<Event>? events}) async {
    final prefs = await SharedPreferences.getInstance();
    if (position != null) {
      await prefs.setString('latest_position', jsonEncode(position.toJson()));
    }
    if (events != null) {
      final eventList = events.map((e) => {
            'title': e.title ?? '',
            'start': e.start?.toIso8601String() ?? '',
            'end': e.end?.toIso8601String() ?? '',
            'location': e.location ?? '',
          }).toList();
      await prefs.setString('latest_events', jsonEncode(eventList));
    }
  }

  @override
  Future<void> onStart(DateTime timestamp, TaskStarter starter) async {
    await dotenv.load(fileName: ".env");
    print('[BACKGROUND LOG] onStart() initiated.'); // LOG
    await _loadSchedule();
    await _subscribeToLocationStream(_currentThreshold);
    try {
      final initialEvents = await _fetchCalendarEvents();
      print('[BACKGROUND LOG] Sending initial calendar data to backend.');
      await _sendCalendarUpdate(initialEvents); // <-- 백엔드 전송 코드 추가
      _lastSentEvents = initialEvents;
      await _saveDataToPrefs(events: _lastSentEvents);
    } catch (e) {
      print('[BACKGROUND LOG] ERROR during initial calendar load: $e'); // LOG
    }
  }

  @override
  Future<void> onRepeatEvent(DateTime timestamp) async {
    print('[BACKGROUND LOG] onRepeatEvent() initiated.'); // LOG
    if (_posSub != null && _posSub!.isPaused && _isWithinSchedule()) {
      _posSub!.resume();
      print('[BACKGROUND LOG] Resuming location stream due to schedule.'); // LOG
    }

    List<Event> currentEvents = [];
    try {
      currentEvents = await _fetchCalendarEvents();
    } catch (e) {
      print('[BACKGROUND LOG] ERROR fetching calendar in onRepeatEvent: $e'); // LOG
    }

    if (!_areEventsEqual(_lastSentEvents, currentEvents)) {
      print('[BACKGROUND LOG] Calendar events have changed. Sending update.'); // LOG
      await _sendCalendarUpdate(currentEvents); // 백엔드 전송
      _lastSentEvents = currentEvents;
      await _saveDataToPrefs(events: currentEvents, position: _lastPos);
    }
    
    if (_posSub != null && !_posSub!.isPaused && !_isWithinSchedule()) {
      _posSub!.pause();
      print('[BACKGROUND LOG] Pausing location stream due to schedule.'); // LOG
    }
  }

  bool _areEventsEqual(List<Event> a, List<Event> b) {
    if (a.length != b.length) return false;
    final aSet = a
        .map((e) => '${e.eventId}|${e.title}|${e.start}|${e.end}|${e.location}')
        .toSet();
    final bSet = b
        .map((e) => '${e.eventId}|${e.title}|${e.start}|${e.end}|${e.location}')
        .toSet();
    return aSet.length == bSet.length && aSet.containsAll(bSet);
  }

  Future<void> _sendCalendarUpdate(List<Event> evs) async {
    final now = DateTime.now();
    final payload = {
      'userId': 'test_user_01',
      'recordTime': now.toIso8601String(),
      'upcomingEvents': evs
          .map(
            (e) => {
              'title': e.title ?? '',
              'startTime': e.start?.toIso8601String() ?? '',
              'endTime': e.end?.toIso8601String() ?? '',
              'location': e.location ?? '',
            },
          )
          .toList(),
    };
    try {
      final res = await http.post(
        Uri.parse('${dotenv.env['SERVER_URL']}/api/v1/calendar'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );
      print('✅ [BACKGROUND LOG] Calendar update result: ${res.statusCode}'); // LOG
    } catch (e) {
      print('❌ [BACKGROUND LOG] Calendar update ERROR: $e'); // LOG
    }
  }

  Future<void> _loadSchedule() async {
    final prefs = await SharedPreferences.getInstance();
    final str = prefs.getString('schedule');
    if (str != null) {
      final Map<String, dynamic> data = jsonDecode(str);
      data.forEach((day, ranges) {
        final wd = _dayKeyToWeekday(day);
        _schedule[wd] = (ranges as List)
            .map(
              (r) => TimeRange(
                int.parse(r.split('-')[0].split(':')[0]) * 60 +
                    int.parse(r.split('-')[0].split(':')[1]),
                int.parse(r.split('-')[1].split(':')[0]) * 60 +
                    int.parse(r.split('-')[1].split(':')[1]),
              ),
            )
            .toList();
      });
    }
  }

  int _dayKeyToWeekday(String key) {
    const map = {
      'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7,
    };
    return map[key] ?? 1;
  }

  bool _isWithinSchedule() {
    if (_schedule.values.every((l) => l.isEmpty)) return true;
    final now = TimeOfDay.now();
    final wd = DateTime.now().weekday;
    final mins = now.hour * 60 + now.minute;
    final List<TimeRange> list = _schedule[wd] ?? <TimeRange>[];
    return list.any((r) => r.contains(mins));
  }

  Future<void> _subscribeToLocationStream(int distanceFilter) async {
    await _posSub?.cancel();
    _posSub = Geolocator.getPositionStream(
      locationSettings: LocationSettings(
        accuracy: LocationAccuracy.medium,
        distanceFilter: distanceFilter,
      ),
    ).listen(_processData);
    print('[BACKGROUND LOG] Resubscribed to location stream: distanceFilter = $distanceFilter m'); // LOG
  }

  @override
  Future<void> _processData(Position p) async {
    // --- ⭐️ 1. 첫 위치 수신 처리 (핵심 버그 수정) ⭐️ ---
    if (_lastPos == null) {
      print('[BACKGROUND LOG] First location received. Saving initial position.');
      _lastPos = p;
      _lastMajorMoveTime = DateTime.now();
      await _sendLocationToBackend(p);
      await _saveDataToPrefs(position: p);
      return; // 첫 위치는 기준점만 잡고 로직 종료
    }

    // 스케줄 시간 외에는 항상 동작하지 않음
    if (!_isWithinSchedule()) {
      if (_posSub != null && !_posSub!.isPaused) {
        _posSub!.pause();
        print('[BACKGROUND LOG] Paused location stream (outside schedule).');
      }
      return;
    } else {
      if (_posSub != null && _posSub!.isPaused) {
        _posSub!.resume();
        print('[BACKGROUND LOG] Resumed location stream (inside schedule).');
      }
    }

    final now = DateTime.now();
    final double distance = Geolocator.distanceBetween(
        _lastPos!.latitude, _lastPos!.longitude, p.latitude, p.longitude);
    
    bool locationSent = false;

    // 현재 임계값(Threshold)에 따라 로직 분기
    switch (_currentThreshold) {
      case 300:
        if (distance >= 300) {
          if (now.difference(_lastMajorMoveTime).inSeconds <= 45) {
            print('[BACKGROUND LOG] Vehicle detected. Switching to 2km threshold.');
            await _sendLocationToBackend(p);
            _currentThreshold = 2000;
            await _subscribeToLocationStream(_currentThreshold);
            _consecutive2kmCount = 1;
            _last2kmMoveTime = now;
          } 
          else {
            print('[BACKGROUND LOG] Standard 300m move detected.');
            await _sendLocationToBackend(p);
          }
          locationSent = true;
        }
        break;

      case 2000:
        if (_last2kmMoveTime != null && now.difference(_last2kmMoveTime!).inMinutes >= 6) {
          print('[BACKGROUND LOG] 2km threshold timeout (6 minutes). Switching to 300m.');
          await _sendLocationToBackend(p);
          _currentThreshold = 300;
          await _subscribeToLocationStream(_currentThreshold);
          _consecutive2kmCount = 0; 
          locationSent = true;
        }
        else if (distance >= 2000) {
          print('[BACKGROUND LOG] 2km move detected within 6 minutes.');
          await _sendLocationToBackend(p);
          _consecutive2kmCount++;
          _last2kmMoveTime = now;

          if (_consecutive2kmCount >= 3) {
            print('[BACKGROUND LOG] 3 consecutive 2km moves. Switching to 10km threshold.');
            _currentThreshold = 10000;
            await _subscribeToLocationStream(_currentThreshold);
            // --- ⭐️ 2. 카운트 리셋 코드 추가 ⭐️ ---
            _consecutive2kmCount = 0;
          }
          locationSent = true;
        }
        break;

      case 10000:
        if (distance >= 10000) {
          print('[BACKGROUND LOG] 10km move detected.');
          await _sendLocationToBackend(p);
          locationSent = true;
        } 
        else if (now.difference(_lastMajorMoveTime).inMinutes >= 20) {
          print('[BACKGROUND LOG] 10km threshold timeout (20 minutes). Switching to 300m.');
          _currentThreshold = 300;
          await _subscribeToLocationStream(_currentThreshold);
          _lastMajorMoveTime = now;
        }
        break;
    }

    // 백엔드 전송이 일어난 모든 경우, 마지막 위치와 시간 정보 업데이트
    if (locationSent) {
      _lastPos = p;
      _lastMajorMoveTime = now;
      await _saveDataToPrefs(position: p);
    }
  }

  Future<void> _sendLocationToBackend(Position p) async {
    final now = DateTime.now();
    final payload = {
      'userId': 'test_user_01',
      'recordTime': now.toIso8601String(),
      'location': {
        'latitude': p.latitude,
        'longitude': p.longitude,
        'timestamp': (p.timestamp ?? now).toIso8601String(),
      },
    };
    try {
      final res = await http.post(
        Uri.parse('${dotenv.env['SERVER_URL']}/api/v1/location'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );
      print('✅ [BACKGROUND LOG] Location sent, result: ${res.statusCode}'); // LOG
    } catch (e) {
      print('❌ [BACKGROUND LOG] Location send ERROR: $e'); // LOG
    }
  }

  Future<List<Event>> _fetchCalendarEvents() async {
    print('---- 🔍 [BACKGROUND LOG] Background fetch calendar started ----'); // LOG
    final plugin = DeviceCalendarPlugin();
    final cals = await plugin.retrieveCalendars();
    if (cals.isSuccess && cals.data != null) {
      print('🗓️ [BACKGROUND LOG] Found ${cals.data!.length} calendars.'); // LOG
      List<Event> list = [];
      for (var cal in cals.data!) {
        print('   -> [BACKGROUND LOG] Processing calendar: ${cal.name}'); // LOG
        if (cal.id != null) {
          final evs = await plugin.retrieveEvents(
            cal.id!,
            RetrieveEventsParams(
              startDate: DateTime.now(),
              endDate: DateTime.now().add(const Duration(days: 7)),
            ),
          );
          if (evs.isSuccess && evs.data != null) {
            print('       => [BACKGROUND LOG] Found ${evs.data!.length} events.'); // LOG
            list.addAll(evs.data!);
          }
        }
      }
      print('✨ [BACKGROUND LOG] Total events fetched: ${list.length}'); // LOG
      return list;
    }
    print('🚨 [BACKGROUND LOG] Calendar retrieval failed.'); // LOG
    return [];
  }

  @override
  Future<void> onDestroy(DateTime timestamp, bool isTimeout) async {
    print('[BACKGROUND LOG] onDestroy() called.'); // LOG
    await _posSub?.cancel();
  }
}