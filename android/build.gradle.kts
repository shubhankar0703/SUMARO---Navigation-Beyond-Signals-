plugins {
    id("com.android.application") version "8.7.3" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.0.21" apply false
}

allprojects {
    val buildBaseDir = File(System.getProperty("user.home"), ".android_builds/SUMARO/${project.name}")
    layout.buildDirectory.set(buildBaseDir)
}
