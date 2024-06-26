$pkg = "matrix"

pip show $pkg > $null 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "$pkg lib is already installed"
    exit 0
} else {
    Write-Host "$pkg isn't installed yet"
}

Write-Host "Initiating the installation process..."

if (Test-Path "./setup.py") {
    Write-Host "setup.py has been found!"
} else {
    Write-Host "setup.py file not found!"
    exit 1
}

python ./setup.py sdist bdist_wheel
Set-Location ./dist
pip install *.whl
Set-Location ..
Remove-Item -Recurse -Force ./dist, ./build, ./matrix.egg-info

pip show $pkg > null 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "$pkg lib has been installed successfully"
} else {
    Write-Host "something bad has happened! try to install manually"
    exit 1
}
