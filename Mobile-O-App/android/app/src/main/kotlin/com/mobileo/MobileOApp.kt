package com.mobileo

import android.app.Application
import android.util.Log

/**
 * Application class. Mirrors iOS MobileOApp.swift @main entry point.
 */
class MobileOApp : Application() {

    override fun onCreate() {
        super.onCreate()
        Log.i("MobileOApp", "Mobile-O Android initialized")
    }
}
