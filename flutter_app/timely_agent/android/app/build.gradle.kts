plugins {
    id("com.android.application")
    id("kotlin-android")
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.timely_agent"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = "27.0.12077973"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_11.toString()
    }

    defaultConfig {
        applicationId = "com.example.timely_agent"
        minSdk = 23
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // 여기에 ProGuard/R8 규칙을 추가합니다.
            isMinifyEnabled = true // '=' 를 사용하고, 변수명이 isMinifyEnabled 입니다.
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro") // 괄호()를 사용합니다.
            // signingConfig = signingConfigs.getByName("debug")
            // signingConfig는 나중에 생성할 본인의 'release' 키를 사용해야 합니다.
            // 아직 릴리즈 키를 만들지 않았다면, 아래 라인은 주석 처리하거나 삭제해도 괜찮습니다.
            // signingConfig signingConfigs.release
        }
    }
}

dependencies {
    implementation("com.google.android.gms:play-services-location:21.2.0")
}

flutter {
    source = "../.."
}