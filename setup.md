curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user | grep login

mv .github github_temp #unhide
mv github_temp .github #rehide

source "$(poetry env info --path)/bin/activate"