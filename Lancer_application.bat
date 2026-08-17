@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   RBSA-ML - Analyse de style des OPCVM Actions
echo   Africapital Management
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERREUR] Python n'est pas trouve. Installe Python 3.10+ depuis python.org
    echo puis relance ce fichier.
    pause
    exit /b 1
)

if not exist "outputs\models\gru_best.pt" (
    echo Aucun modele entraine trouve : premiere execution.
    echo Installation des dependances...
    python -m pip install -r requirements.txt --quiet
    echo.
    echo Entrainement du modele sur les donnees reelles ^(environ 15 minutes^)...
    echo Tu peux laisser cette fenetre ouverte et faire autre chose en attendant.
    python scripts\run_pipeline.py
    echo.
) else (
    echo Modele deja entraine trouve, verification des dependances...
    python -m pip install -r requirements.txt --quiet
)

echo Lancement de l'application ^(elle va s'ouvrir dans ton navigateur^)...
echo Pour arreter l'application, ferme cette fenetre ou appuie sur Ctrl+C.
echo.
streamlit run app\streamlit_app.py

pause
