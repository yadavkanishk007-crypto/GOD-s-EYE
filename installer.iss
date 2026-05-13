[Setup]
AppName=GOD's EYE
AppVersion=1.0.0
AppPublisher=Indian Urban Intelligence
AppPublisherURL=https://github.com/urban-intelligence
DefaultDirName={autopf}\GodsEye
DefaultGroupName=GOD's EYE Command Center
OutputDir=Output
OutputBaseFilename=GodsEye_Setup
Compression=lzma
SolidCompression=yes
LicenseFile=LICENSE.txt
InfoAfterFile=README_INSTALL.txt
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Copy the compiled PyInstaller output
Source: "dist\GodsEye\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Copy the config file and model
Source: "config.yaml"; DestDir: "{app}"; Flags: ignoreversion
Source: "yolov8n.pt"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "yolov8n.onnx"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Dirs]
; Create data directories for persistence
Name: "{app}\data"; Permissions: users-modify
Name: "{app}\data\events"; Permissions: users-modify
Name: "{app}\data\models"; Permissions: users-modify
Name: "{app}\data\logs"; Permissions: users-modify

[Icons]
Name: "{group}\GOD's EYE"; Filename: "{app}\GodsEye.exe"
Name: "{autodesktop}\GOD's EYE"; Filename: "{app}\GodsEye.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\GodsEye.exe"; Description: "{cm:LaunchProgram,GOD's EYE}"; Flags: nowait postinstall skipifsilent
