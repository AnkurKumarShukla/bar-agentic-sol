# import asyncio
# import websockets
# import json

# # Function to handle the chat client
# async def chat():
#     try:
#         async with websockets.connect('ws://127.0.0.1:8000/ws/chat') as websocket:
#             while True:
#                 try:
#                     # Prompt the user for a message
#                     # message = input("Enter message: ")
#                     payload = {
#                         "dataset": "golden_data",
#                         "thread_id":"10017",
#                         "user_id":"1",
#                         "message":"Fetch the latest news sentiment for Lloyds Bank and summarise if the market outlook is positive or negative."
#                     }
#                     # Send the message to the server
#                     await websocket.send(json.dumps(payload))
#                     # Receive a message from the server
#                     response = await websocket.recv()
#                     print(f"Received: {response}")
#                     data = json.loads(response)
#                     msg_type = data.get("type")

#                     if msg_type == "chunk":
#                         final_state = data.get("state", {})
#                         print("\n🧩 Received chunk:")
#                     elif msg_type == "final":
#                         print("✅ Conversation completed successfully.")
#                         await asyncio.sleep(5)  # let backend finish cleanly
#                         # break
#                     elif msg_type == "error":
#                         print(f"❌ Error: {data.get('message')}")
#                         break
#                     else:
#                         print(f"⚙️ Unknown message type: {data}")
#                         break

#                 except websockets.ConnectionClosedOK:
#                     print("🔚 Server closed connection cleanly.")
#                     break
#                 except websockets.ConnectionClosedError as e:
#                     print(f"⚠️ Connection closed with error: {e}")
#                     break

#     except Exception as e:
#         print(f"❌ Exception occurred: {e}")

# # Run the client
# if __name__ == "__main__":
#     asyncio.run(chat())

import asyncio
import websockets
import json

async def chat():
    try:
        async with websockets.connect('ws://127.0.0.1:8000/ws/chat') as websocket:
            payload_sent = False  # flag to send payload only once
            finished = False      # flag to track if final received

            while True:
                try:
                    # Send the payload only once
                    if not payload_sent:
                        payload = {
                            "dataset": "golden_data",
                            "thread_id": "10020",
                            "user_id": "1",
                            "message": "Fetch the latest news sentiment for Lloyds Bank and summarise if the market outlook is positive or negative."
                        }
                        await websocket.send(json.dumps(payload))
                        payload_sent = True

                    # Receive a message from the server
                    response = await websocket.recv()
                    data = json.loads(response)
                    msg_type = data.get("type")

                    if msg_type == "chunk":
                        final_state = data.get("state", {})
                        print("\n🧩 Received chunk:")
                        # You can process chunk here

                    elif msg_type == "final":
                        print("✅ Conversation completed successfully.")
                        finished = True
                        # Stop sending new payloads but keep connection alive
                        # You can now trigger further processing here

                    elif msg_type == "error":
                        print(f"❌ Error: {data.get('message')}")
                        break

                    else:
                        print(f"⚙️ Unknown message type: {data}")
                        break

                    # After final, optionally wait for more server events without resending
                    if finished:
                        await asyncio.sleep(1)  # small sleep to prevent busy loop

                except websockets.ConnectionClosedOK:
                    print("🔚 Server closed connection cleanly.")
                    break
                except websockets.ConnectionClosedError as e:
                    print(f"⚠️ Connection closed with error: {e}")
                    break

    except Exception as e:
        print(f"❌ Exception occurred: {e}")


if __name__ == "__main__":
    asyncio.run(chat())
