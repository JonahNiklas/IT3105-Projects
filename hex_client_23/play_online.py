# Import and initialize your own actor
from hex_client_23.ActorClient import ActorClient
from project2.hex_actor import HexActor

actor = HexActor()

# Import and override the `handle_get_action` hook in ActorClient


class MyClient(ActorClient):
    def handle_get_action(self, state):
        row, col = actor.get_action(state)  # Your logic
        return row, col


# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient(qualify=False)
    client.run()
