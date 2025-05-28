# Pushing to GitHub

This guide will help you push your Face Recognition Attendance System to GitHub.

## Prerequisites

- [Git](https://git-scm.com/downloads) installed on your computer
- A [GitHub](https://github.com/) account

## Steps to Push to GitHub

1. **Create a new repository on GitHub**
   - Go to [GitHub](https://github.com/) and sign in
   - Click the '+' icon in the top right and select 'New repository'
   - Name your repository (e.g., "face-recognition-attendance-system")
   - Add a description (optional)
   - Choose public or private visibility
   - Do NOT initialize with README, .gitignore, or license (we already have these)
   - Click 'Create repository'

2. **Initialize Git in your local project**
   - Open Command Prompt or PowerShell
   - Navigate to your project directory:
     ```
     cd "c:\Users\saura\Downloads\face_recognition_system upgraded"
     ```
   - Initialize Git:
     ```
     git init
     ```

3. **Add your files to Git**
   - Add all files:
     ```
     git add .
     ```
   - Commit the files:
     ```
     git commit -m "Initial commit"
     ```

4. **Link your local repository to GitHub**
   - Add the remote repository URL (replace with your actual repository URL):
     ```
     git remote add origin https://github.com/yourusername/face-recognition-attendance-system.git
     ```
   - Push your code to GitHub:
     ```
     git push -u origin master
     ```
     (If your default branch is 'main' instead of 'master', use 'main' in the command)

5. **Verify your repository**
   - Go to your GitHub repository URL to confirm that all files have been uploaded

## Updating Your Repository

After making changes to your project:

1. Add the changed files:
   ```
   git add .
   ```

2. Commit the changes:
   ```
   git commit -m "Description of changes"
   ```

3. Push to GitHub:
   ```
   git push
   ```

## Additional Resources

- [GitHub Documentation](https://docs.github.com/en)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)