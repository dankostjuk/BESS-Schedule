FROM gradle:8-jdk21-alpine AS build
WORKDIR /app
COPY . .
RUN gradle installDist --no-daemon

FROM eclipse-temurin:21-jre-alpine
WORKDIR /app
COPY --from=build /app/build/install/overview/ .
COPY --from=build /app/data/ data/
EXPOSE 8080
CMD ["bin/overview"]
