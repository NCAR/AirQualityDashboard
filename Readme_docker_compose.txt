To run the air quality dashboard using docker compose:

	ksampson@compass:/gis/air_quality/app/AirQualityDashboard$ docker compose up
	
	
To build the containers:
	Navigate to the directory containing the docker-compose.yml file and use:
		docker compose build
		
	To run the container:
		docker compose up
		
	To run the container in background:
		$ docker compose up --detach
		
	To bring down the container:
		docker compose down
		
	To check the running services:
		docker compose ps
		
	
Found at:
	https://gist.github.com/ngtrieuvi92/f41b7ecaf6eb6ee18b927100d155bc97
	
	
	# Remove all exited container
	docker rm $(docker ps -q -f status=exited)

	# Remove none tag images (image with tag <none>)
	docker images | grep "<none>" | awk '{print $3}' |xargs docker rmi -f

	# Remove all images that is not using by any running container
	# note: docker ps --format {{.Image} -> List all images of running container then set it as grep pattern
	docker images --format {{.Repository}}:{{.Tag}} | grep -vFf <(docker ps --format {{.Image}}) | xargs docker rmi -f