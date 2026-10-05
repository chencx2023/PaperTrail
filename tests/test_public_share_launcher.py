import unittest
import json
import tempfile
import inspect
from pathlib import Path
import httpx
from scripts import share_papertrail as launcher
from scripts.share_papertrail import tunnel_command, find_public_url


class PublicShareLauncherTest(unittest.TestCase):
    def test_ssh_fallback_has_no_user_credentials_or_account(self):
        command = tunnel_command("localhost.run", Path("cloudflared.exe"), 8787, Path(".tooling/share"))
        self.assertIn("nokey@localhost.run", command)
        self.assertIn("80:127.0.0.1:8787", command)
        self.assertIn("PubkeyAuthentication=no", command)
        self.assertIn("IdentityAgent=none", command)
        self.assertIn("BatchMode=yes", command)

    def test_only_expected_provider_hostnames_become_links(self):
        self.assertEqual(find_public_url("cloudflare", "url https://example.trycloudflare.com"), "https://example.trycloudflare.com")
        self.assertEqual(find_public_url("localhost.run", "example.lhr.life tunneled with tls termination, https://example.lhr.life"), "https://example.lhr.life")
        self.assertIsNone(find_public_url("localhost.run", "docs https://attacker.example.com"))

    def test_management_banner_is_not_a_public_endpoint(self):
        banner = "To set up and manage custom domains go to https://admin.localhost.run/\nhttps://localhost.run/docs/\n"
        self.assertIsNone(find_public_url("localhost.run", banner))
        announcement = "a9a8cc686bfebc.lhr.life tunneled with tls termination, https://a9a8cc686bfebc.lhr.life\n"
        self.assertEqual(find_public_url("localhost.run", banner + announcement), "https://a9a8cc686bfebc.lhr.life")
        self.assertIsNone(find_public_url("localhost.run", "tunneled with tls termination, https://example.lhr.life.example.com"))

    def test_rotated_domain_replaces_the_first_announced_domain(self):
        output = "old.lhr.life tunneled with tls termination, https://old.lhr.life\n"
        output += "new.lhr.life tunneled with tls termination, https://new.lhr.life\n"
        output += "manage domains https://admin.localhost.run/\n"
        self.assertEqual(find_public_url("localhost.run", output), "https://new.lhr.life")

    def test_provider_no_tunnel_response_is_not_public_readiness(self):
        probe = getattr(launcher, "public_access_probe", None)
        self.assertTrue(callable(probe), "Public readiness must check the actual HTTPS endpoint.")
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(503, text="<h1>no tunnel here :(</h1>")
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            self.assertFalse(probe(client, "https://example.lhr.life", "p" * 48))
        self.assertEqual(len(requests), 1)
        self.assertNotIn("authorization", requests[0].headers)

    def test_public_probe_checks_our_gate_and_same_workspace_worker(self):
        probe = getattr(launcher, "public_access_probe", None)
        self.assertTrue(callable(probe), "Public readiness must check the actual HTTPS endpoint.")
        requests = []
        def respond(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(401, json={"code": "PUBLIC_ACCESS_REQUIRED"})
            return httpx.Response(200, json={"worker_status": "running", "shared_workspace": True})
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            self.assertTrue(probe(client, "https://example.lhr.life", "p" * 48))
        self.assertEqual(requests[1].url.path, "/api/runtime")
        self.assertEqual(requests[1].headers["authorization"], "Bearer " + "p" * 48)
        self.assertTrue(all(request.method == "GET" for request in requests))

    def test_wrong_gateway_or_stopped_worker_is_not_ready(self):
        probe = getattr(launcher, "public_access_probe", None)
        self.assertTrue(callable(probe), "Public readiness must check the actual HTTPS endpoint.")
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="another site"))) as client:
            self.assertFalse(probe(client, "https://example.lhr.life", "p" * 48))
        def stopped(request):
            if request.url.path == "/api/health":
                return httpx.Response(401, json={"code": "PUBLIC_ACCESS_REQUIRED"})
            return httpx.Response(200, json={"worker_status": "stopped", "shared_workspace": True})
        with httpx.Client(transport=httpx.MockTransport(stopped)) as client:
            self.assertFalse(probe(client, "https://example.lhr.life", "p" * 48))

    def test_alive_process_can_have_inactive_public_link(self):
        update = getattr(launcher, "update_public_availability", None)
        self.assertTrue(callable(update), "An alive process must not keep a broken public link active.")
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "access.json"
            record.write_text(json.dumps({"pid": 123, "active": True, "share_url": "saved-link"}), encoding="utf-8")
            update(record, 123, False)
            data = json.loads(record.read_text(encoding="utf-8"))
            self.assertFalse(data["active"])
            self.assertEqual(data["share_url"], "saved-link")
            self.assertIn("last_checked_at", data)
            update(record, 999, True)
            self.assertFalse(json.loads(record.read_text(encoding="utf-8"))["active"])

    def test_verified_rotation_updates_both_public_and_credential_links(self):
        update = launcher.update_public_availability
        self.assertIn("public_url", inspect.signature(update).parameters, "Verified rotations must replace the saved public link.")
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "access.json"
            token = "p" * 48
            record.write_text(json.dumps({"pid": 123, "active": False, "access_token": token, "public_url": "https://old.lhr.life", "share_url": "old-link"}), encoding="utf-8")
            update(record, 123, True, public_url="https://new.lhr.life")
            data = json.loads(record.read_text(encoding="utf-8"))
            self.assertTrue(data["active"])
            self.assertEqual(data["public_url"], "https://new.lhr.life")
            self.assertEqual(data["share_url"], "https://new.lhr.life/?access=" + token)
            self.assertEqual(data["access_token"], token)
            update(record, 123, False, public_url="https://unverified.lhr.life")
            self.assertEqual(json.loads(record.read_text(encoding="utf-8"))["public_url"], "https://new.lhr.life")


if __name__ == "__main__":
    unittest.main()
