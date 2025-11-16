// frontend/src/components/Intersection.jsx
import React, { useEffect, useRef } from 'react';
import './Intersection.css';

const Intersection = ({ metrics }) => {
  const canvasRef = useRef(null);
  const vehiclesRef = useRef([]);
  const metricsRef = useRef(metrics);

  useEffect(() => {
    metricsRef.current = metrics;
  }, [metrics]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const laneWidth = 80;
    const vehicleSize = 10;

    const lanePositions = {
      north: { x: centerX + 10, y: 0 },
      south: { x: centerX - 10, y: height },
      east: { x: width, y: centerY + 10 },
      west: { x: 0, y: centerY - 10 },
    };

    const stopLines = {
        north: centerY - laneWidth / 2,
        south: centerY + laneWidth / 2,
        east: centerX + laneWidth / 2,
        west: centerX - laneWidth / 2,
    };

    class Vehicle {
      constructor(lane) {
        this.lane = lane;
        this.x = lanePositions[lane].x;
        this.y = lanePositions[lane].y;
        this.speed = 2;
        this.size = vehicleSize;
        this.color = `hsl(${Math.random() * 360}, 100%, 50%)`;
      }

      draw(ctx) {
        ctx.fillStyle = this.color;
        ctx.fillRect(this.x - this.size / 2, this.y - this.size / 2, this.size, this.size);
      }

      update() {
        const currentMetrics = metricsRef.current;
        if (!currentMetrics) return;

        const isNSGreen = currentMetrics.current_phase === 'NORTH_SOUTH';
        const isEWGreen = currentMetrics.current_phase === 'EAST_WEST';

        let isRedLight = false;
        let stopLine = 0;

        switch (this.lane) {
          case 'north':
            isRedLight = !isNSGreen;
            stopLine = stopLines.north;
            if (!isRedLight || this.y > stopLine) {
                this.y += this.speed;
            }
            break;
          case 'south':
            isRedLight = !isNSGreen;
            stopLine = stopLines.south;
            if (!isRedLight || this.y < stopLine) {
                this.y -= this.speed;
            }
            break;
          case 'east':
            isRedLight = !isEWGreen;
            stopLine = stopLines.east;
            if (!isRedLight || this.x < stopLine) {
                this.x -= this.speed;
            }
            break;
          case 'west':
            isRedLight = !isEWGreen;
            stopLine = stopLines.west;
            if (!isRedLight || this.x > stopLine) {
                this.x += this.speed;
            }
            break;
          default:
            break;
        }
      }
    }

    const updateVehicles = () => {
        const currentMetrics = metricsRef.current;
        if (!currentMetrics) return;

        const laneQueue = {
            north: currentMetrics.north.queue_length,
            south: currentMetrics.south.queue_length,
            east: currentMetrics.east.queue_length,
            west: currentMetrics.west.queue_length,
        };

        Object.keys(laneQueue).forEach(lane => {
            const currentLaneVehicles = vehiclesRef.current.filter(v => v.lane === lane);
            const diff = laneQueue[lane] - currentLaneVehicles.length;

            if (diff > 0) {
                for (let i = 0; i < diff; i++) {
                    vehiclesRef.current.push(new Vehicle(lane));
                }
            }
        });

        vehiclesRef.current = vehiclesRef.current.filter(v => 
            v.x >= -v.size && v.x <= width + v.size && v.y >= -v.size && v.y <= height + v.size
        );
    };
    
    const intervalId = setInterval(updateVehicles, 1000);

    let animationFrameId;
    const animate = () => {
      ctx.fillStyle = '#E5E7EB'; // Light gray background
      ctx.fillRect(0, 0, width, height);

      // Roads
      ctx.fillStyle = '#374151'; // Dark gray for roads
      ctx.fillRect(0, centerY - laneWidth / 2, width, laneWidth);
      ctx.fillRect(centerX - laneWidth / 2, 0, laneWidth, height);

      // Lane markings
      ctx.strokeStyle = '#FBBF24'; // Yellow for lane markings
      ctx.lineWidth = 3;
      ctx.setLineDash([20, 20]);
      
      // Vertical lanes
      ctx.beginPath();
      ctx.moveTo(centerX, 0);
      ctx.lineTo(centerX, centerY - laneWidth/2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(centerX, height);
      ctx.lineTo(centerX, centerY + laneWidth/2);
      ctx.stroke();

      // Horizontal lanes
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(centerX - laneWidth/2, centerY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(width, centerY);
      ctx.lineTo(centerX + laneWidth/2, centerY);
      ctx.stroke();

      ctx.setLineDash([]);

      // Traffic Lights
      const drawTrafficLight = (x, y, isGreen) => {
        ctx.fillStyle = isGreen ? '#10B981' : '#EF4444'; // Green/Red
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, 2 * Math.PI);
        ctx.fill();
      };

      if(metricsRef.current) {
        const isNSGreen = metricsRef.current.current_phase === 'NORTH_SOUTH';
        drawTrafficLight(centerX - 50, centerY - 50, isNSGreen); // NW
        drawTrafficLight(centerX + 50, centerY + 50, isNSGreen); // SE
        drawTrafficLight(centerX + 50, centerY - 50, !isNSGreen); // NE
        drawTrafficLight(centerX - 50, centerY + 50, !isNSGreen); // SW
      }

      // Update and draw vehicles
      vehiclesRef.current.forEach(v => {
        v.update();
        v.draw(ctx);
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      clearInterval(intervalId);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return <canvas ref={canvasRef} width={600} height={600} className="intersection-canvas" />;
};

export default Intersection;
